"""The student: a supercombo replacement built for the AX630C NPU (Pulsar2 AX620E target, NPU1, U16).

Same inputs as the edgepilot core (input_imgs / big_input_imgs [1,12,128,256] uint8 YUV6 of frames t-4 and t,
desire [1,25,8], features_buffer [1,24,512], traffic_convention [1,2]) and the outputs of the CPU-heads core
(cpu_heads.py): out_hidden [1,512] fed back as the feature history, and per head the last ReLU feature hx_<head>
whose final linear layer runs on the CPU in fp32.

What makes it NPU-friendly (from the AX620E profile of the current core: depthwise convs 26 % and the
tiling Slice/assign 26 % of the vision cycles, GELU 4.5 %):
  - only dense 3x3 / 1x1 convs + ReLU (BatchNorm and RepVGG branches folded into one conv for export);
  - the resolution falls fast, so few large feature maps go through the NPU's tiling;
  - no LayerNorm, GELU, layer scale or attention; the temporal mixing is a few small FCs over the last 9 states;
  - every head ends in ReLU features on the NPU and an fp32 linear layer on the CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

HEADS = {  # name: (feature width, output width on the CPU), the CPU-heads core's layout
    "meta": (64, 55), "desire_pred": (32, 32), "pose": (32, 12), "wide_from_device": (32, 6),
    "road_transform": (32, 12), "lanes": (64, 528), "lane_prob": (16, 8), "road_edges": (32, 264),
    "lead": (64, 144), "lead_prob": (16, 3),
}
POLICY_HEADS = {"plan": (256, 990), "desire_state": (32, 8)}


class RepConv(nn.Module):
    """3x3 conv + BN, a parallel 1x1 conv + BN and (when shapes allow) an identity BN, then ReLU.
    fuse() folds it into one 3x3 conv with bias for export. With qat set (quant_sim), the forward folds the branches
    the same way (BN as its running statistics, which then stay fixed) and convolves with the S8-rounded weights."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.cin, self.cout, self.stride = cin, cout, stride
        self.k3 = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout))
        self.k1 = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, 0, bias=False), nn.BatchNorm2d(cout))
        self.idn = nn.BatchNorm2d(cout) if cin == cout and stride == 1 else None
        self.fused = None
        self.qat = False

    def forward(self, x):
        if self.fused is not None:
            return F.relu(self.fused(x))
        if self.qat:
            w, b = self.folded()
            return F.relu(F.conv2d(x, fake_quant_s8(w), b, self.stride, 1))
        y = self.k3(x) + self.k1(x)
        if self.idn is not None:
            y = y + self.idn(x)
        return F.relu(y)

    @staticmethod
    def _fold(w, bn):
        std = (bn.running_var + bn.eps).sqrt()
        return w * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * bn.weight / std

    def folded(self):
        """The one 3x3 conv's weight and bias (differentiable)."""
        w3, b3 = self._fold(self.k3[0].weight, self.k3[1])
        w1, b1 = self._fold(F.pad(self.k1[0].weight, [1, 1, 1, 1]), self.k1[1])
        w, b = w3 + w1, b3 + b1
        if self.idn is not None:
            wi = torch.zeros_like(w3)
            wi[torch.arange(self.cout), torch.arange(self.cout), 1, 1] = 1.0
            wi, bi = self._fold(wi, self.idn)
            w, b = w + wi, b + bi
        return w, b

    @torch.no_grad()
    def fuse(self):
        if self.fused is not None:
            return
        w, b = self.folded()
        self.fused = nn.Conv2d(self.cin, self.cout, 3, self.stride, 1, bias=True).to(w.device)
        self.fused.weight.copy_(w)
        self.fused.bias.copy_(b)
        del self.k3, self.k1, self.idn
        self.idn = None


class Backbone(nn.Module):
    """YUV6 x 2 frames [12,128,256] -> [C, 4, 8]."""

    def __init__(self, widths=(32, 48, 96, 192, 384), depths=(1, 2, 3, 4, 2), strides=(2, 2, 2, 2, 2)):
        super().__init__()
        layers, cin = [], 12
        for i, (w, d) in enumerate(zip(widths, depths)):
            layers.append(RepConv(cin, w, stride=strides[i]))
            layers += [RepConv(w, w) for _ in range(d - 1)]
            cin = w
        self.body = nn.Sequential(*layers)
        self.out_ch = cin

    def forward(self, x):
        return self.body(x)

    def first_conv(self) -> nn.Conv2d:
        """The fused first conv, which takes the input scale at export."""
        return self.body[0].fused

    def is_fused(self) -> bool:
        return self.body[0].fused is not None


def fold_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """conv followed by bn (running statistics) as one conv with bias."""
    std = (bn.running_var + bn.eps).sqrt()
    f = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=True).to(conv.weight.device)
    with torch.no_grad():
        f.weight.copy_(conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1))
        b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
        f.bias.copy_(bn.bias + (b - bn.running_mean) * bn.weight / std)
    return f


# BT.601 YUV -> RGB (0..255, U and V centred at 128) and torchvision's ImageNet normalisation
YUV2RGB = torch.tensor([[1.0, 0.0, 1.402], [1.0, -0.344, -0.714], [1.0, 1.772, 0.0]])
IMNET_MEAN, IMNET_STD = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])


class R18Backbone(nn.Module):
    """torchvision's ImageNet ResNet-18 on YUV6 x 2 frames [12,128,256] -> [C, 4, 8] (backbone="r18").

    conv1 (7x7, stride 2) runs on the half-resolution planes: the current frame's channels get the pretrained RGB
    kernel through the YUV->RGB matrix and the ImageNet normalisation (each Y phase a quarter of the Y weight, so their
    mean is the luminance), the previous frame's start at zero (the pretrained single-frame net, which learns the motion
    from there). Then maxpool and layer1..layer3 (8x16) and a stride-2 RepConv to 4x8; with layers=4, layer4 instead.
    fuse() folds every BN into its conv for export and QAT."""

    def __init__(self, layers=3, pretrained=True, in_scale=1.0 / 64.0):
        super().__init__()
        import torchvision
        tv = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.conv1 = nn.Conv2d(12, 64, 7, 2, 3, bias=True)
        with torch.no_grad():
            w = tv.conv1.weight / (255.0 * IMNET_STD).reshape(1, 3, 1, 1)      # per raw 0..255 RGB unit
            wyuv = torch.einsum("ocij,cd->odij", w, YUV2RGB)                   # per Y, U-128, V-128
            k = torch.zeros(64, 12, 7, 7)
            k[:, 6:10] = wyuv[:, 0:1] / 4
            k[:, 10:12] = wyuv[:, 1:3]
            self.conv1.weight.copy_(k / in_scale)                               # the input arrives times in_scale
            self.conv1.bias.copy_(-(tv.conv1.weight * (IMNET_MEAN / IMNET_STD).reshape(1, 3, 1, 1)).sum((1, 2, 3))
                                  - 128.0 * wyuv[:, 1:3].sum((1, 2, 3)))
        self.bn1, self.maxpool = tv.bn1, tv.maxpool
        self.layers = nn.Sequential(*[getattr(tv, f"layer{i}") for i in range(1, layers + 1)])
        self.down = RepConv(256, 256, stride=2) if layers == 3 else None
        self.out_ch = 256 if layers == 3 else 512
        self.fused_bn = False

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layers(self.maxpool(x))
        return self.down(x) if self.down is not None else x

    @torch.no_grad()
    def fuse_bn(self):
        """Fold every BN into its conv (the RepConv folds with Student.fuse())."""
        if self.fused_bn:
            return
        self.conv1, self.bn1 = fold_bn(self.conv1, self.bn1), nn.Identity()
        for layer in self.layers:
            for blk in layer:
                blk.conv1, blk.bn1 = fold_bn(blk.conv1, blk.bn1), nn.Identity()
                blk.conv2, blk.bn2 = fold_bn(blk.conv2, blk.bn2), nn.Identity()
                if blk.downsample is not None:
                    blk.downsample = nn.Sequential(fold_bn(blk.downsample[0], blk.downsample[1]))
        self.fused_bn = True

    def first_conv(self) -> nn.Conv2d:
        return self.conv1

    def is_fused(self) -> bool:
        """Fused for export (BN folded and the RepConv merged); QAT folds only the BN."""
        return self.fused_bn and (self.down is None or self.down.fused is not None)


def mlp(cin, cout, hidden=None, depth=1):
    hidden = hidden or cout
    layers, c = [], cin
    for _ in range(depth - 1):
        layers += [nn.Linear(c, hidden), nn.ReLU()]
        c = hidden
    layers += [nn.Linear(c, cout), nn.ReLU()]
    return nn.Sequential(*layers)


class Student(nn.Module):
    def __init__(self, widths=(32, 48, 96, 192, 384), depths=(1, 2, 3, 4, 2), strides=(2, 2, 2, 2, 2), neck_ch=64,
                 feat=1024, hidden=512, policy_tokens=9, policy_width=512, head_width=512, temporal_heads=(),
                 backbone="repvgg", r18_layers=3, pretrained=True, lane_policy=False, hist_rank=0, vis_rank=0,
                 lead_cal=False):
        """temporal_heads: vision heads (e.g. lead, lead_prob) whose MLP also sees the temporal features (the
        history), not just the current frames' trunk. backbone: "repvgg" (Backbone) or "r18" (R18Backbone, ImageNet
        ResNet-18; the output grid is 4x8 either way). lane_policy: the temporal MLP also gets the lanes' and road
        edges' head features, so the plan can follow the perceived road geometry (which generalises to new roads)
        instead of the scene. hist_rank r: each history token goes through one shared 512 -> r projection before the
        temporal MLP; vis_rank r: vis_fc is a rank-r product (both cut the FC weights the NPU streams per frame;
        lowrank.py makes them from a full-rank checkpoint). lead_cal: a linear map of the lead features added to the
        lead_prob features, so that one dead lead_prob feature carries relu((x - x0) / s) of the lead's own distance
        (lead_cal.py sets it and the CPU head's weight on it: a far-lead logit shift, no change in the output layout)."""
        super().__init__()
        self.backbone = Backbone(widths, depths, strides) if backbone == "repvgg" else R18Backbone(r18_layers, pretrained)
        hw = (128 // int(torch.tensor(strides).prod())) * (256 // int(torch.tensor(strides).prod()))
        # both towers share the backbone weights (batch of 2), as the master core does
        self.neck = nn.Sequential(nn.Conv2d(self.backbone.out_ch, neck_ch, 1), nn.ReLU())
        self.vis_fc = (nn.Sequential(nn.Linear(2 * neck_ch * hw, vis_rank, bias=False), nn.Linear(vis_rank, feat), nn.ReLU())
                       if vis_rank else mlp(2 * neck_ch * hw, feat))
        self.summarizer = nn.Sequential(nn.Linear(feat, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.vision_trunk = mlp(feat, head_width)
        self.temporal_heads = tuple(temporal_heads)
        self.heads = nn.ModuleDict({k: mlp(head_width + (policy_width if k in self.temporal_heads else 0), w)
                                    for k, (w, _) in HEADS.items()})
        self.policy_tokens = policy_tokens
        self.desire_enc = mlp(25 * 8, 64)
        self.traffic_enc = mlp(2, 16)
        self.lane_policy = lane_policy
        assert not (lane_policy and {"lanes", "road_edges"} & set(temporal_heads))
        self.hist_proj = nn.Linear(hidden, hist_rank, bias=False) if hist_rank else None
        self.temporal = mlp((policy_tokens - 1) * (hist_rank or hidden) + hidden + 64 + 16
                            + (HEADS["lanes"][0] + HEADS["road_edges"][0] if lane_policy else 0), policy_width, policy_width, depth=2)
        self.policy_heads = nn.ModuleDict({k: mlp(policy_width, w, 256, depth=2) for k, (w, _) in POLICY_HEADS.items()})
        # the fp32 linear layers the CPU runs (part of training, exported to the heads file)
        self.cpu_heads = nn.ModuleDict({k: nn.Linear(w, o) for k, (w, o) in {**HEADS, **POLICY_HEADS}.items()})
        # input scale only, no mean subtraction: the BN after the first conv takes the offset, and fuse() folds
        # the scale into that conv's weights (a Sub/Div on the 12x128x256 inputs cost 0.3M cycles on AX620E)
        self.register_buffer("in_scale", torch.tensor(1.0 / 64.0))
        self.lead_cal = nn.Linear(HEADS["lead"][0], HEADS["lead_prob"][0]) if lead_cal else None

    def features(self, input_imgs, big_input_imgs):
        x = torch.cat([input_imgs, big_input_imgs], 0).float() * self.in_scale
        f = self.neck(self.backbone(x))                       # [2, C, 4, 8]
        f = torch.cat([f[0:1].flatten(1), f[1:2].flatten(1)], 1)
        return self.vis_fc(f)

    def forward(self, input_imgs, big_input_imgs, desire, features_buffer, traffic_convention):
        b = self.features(input_imgs, big_input_imgs)
        # L2-normalised like the master's hidden state; on a [B, 1, 512] view: AX620E tiles LpNorm only in 3D/4D
        hidden = F.normalize(self.summarizer(b).unsqueeze(1), dim=-1).squeeze(1)
        hx = self.head_features(b, hidden, features_buffer, desire, traffic_convention)
        return {"out_hidden": hidden, **{"hx_" + k: v for k, v in hx.items()}}

    def head_features(self, feat, hidden, features_buffer, desire, traffic_convention) -> dict:
        """Per head its last ReLU feature (the CPU's fp32 layer makes the output): vision heads from the current
        frames' trunk (temporal_heads: and the temporal features), policy heads from the temporal features."""
        hist = features_buffer[:, -(self.policy_tokens - 1):]
        hist = (self.hist_proj(hist) if self.hist_proj is not None else hist).flatten(1)
        trunk = self.vision_trunk(feat)
        hx = {k: m(trunk) for k, m in self.heads.items() if k not in self.temporal_heads}
        lane = [hx["lanes"], hx["road_edges"]] if self.lane_policy else []
        p = self.temporal(torch.cat([hist, hidden, self.desire_enc(desire.flatten(1)),
                                     self.traffic_enc(traffic_convention), *lane], 1))
        hx.update({k: self.heads[k](torch.cat([trunk, p], 1)) for k in self.temporal_heads})
        hx = {k: hx[k] for k in self.heads}                                  # the heads' order, as exported
        if self.lead_cal is not None:
            hx["lead_prob"] = hx["lead_prob"] + F.relu(self.lead_cal(hx["lead"]))
        hx.update({k: m(p) for k, m in self.policy_heads.items()})
        return hx

    def outputs(self, hx: dict) -> dict:
        """The CPU's fp32 layers: features -> the heads' openpilot outputs."""
        return {k: self.cpu_heads[k](hx["hx_" + k]) for k in self.cpu_heads}

    def fuse(self):
        if isinstance(self.backbone, R18Backbone):
            self.backbone.fuse_bn()
        for m in self.modules():
            if isinstance(m, RepConv):
                m.fuse()
        if float(self.in_scale) != 1.0:
            with torch.no_grad():
                self.backbone.first_conv().weight.mul_(self.in_scale)
                self.in_scale.fill_(1.0)
        return self


class ExportWrapper(nn.Module):
    """ONNX export: the NPU part, outputs in a fixed order."""

    def __init__(self, s: Student):
        super().__init__()
        self.s = s

    def forward(self, input_imgs, big_input_imgs, desire, features_buffer, traffic_convention):
        o = self.s(input_imgs, big_input_imgs, desire, features_buffer, traffic_convention)
        return tuple(o[k] for k in ["out_hidden"] + ["hx_" + k for k in [*HEADS, *POLICY_HEADS]])


def export_onnx(s: Student, path: str):
    if any(hasattr(m, "parametrizations") for m in s.modules()):
        raise ValueError("export a student loaded with load_student(), not one with QAT parametrizations")
    s = s.eval().fuse()
    names = ["out_hidden"] + ["hx_" + k for k in [*HEADS, *POLICY_HEADS]]
    args = (torch.zeros(1, 12, 128, 256), torch.zeros(1, 12, 128, 256), torch.zeros(1, 25, 8), torch.zeros(1, 24, 512),
            torch.tensor([[1.0, 0.0]]))
    torch.onnx.export(ExportWrapper(s), args, path, opset_version=17, dynamo=False,
                      input_names=["input_imgs", "big_input_imgs", "desire", "features_buffer", "traffic_convention"],
                      output_names=names)


def fake_quant_s8(w: torch.Tensor) -> torch.Tensor:
    """Pulsar2's weights: S8 per output channel, scale max|w| / 127.5, codes rounded and clipped to [-128, 127];
    straight-through gradient. These are exactly the int8 codes and scales in a Pulsar2 6.0 AX620E build
    (<build>/quant/quant_axmodel.onnx, every layer of LRA; only dead channels with |w| ~ 1e-14 differ, set to 0 there).
    The earlier scale max|w| / 127 put 3-15 % of the codes one step off: harmless for a float-trained student, but a
    QAT student's weights sit on its grid, so on the NPU it lost the QAT gain (Pulsar2 plan 17.1 vs S8 sim 15.4 cm)."""
    s = w.detach().abs().flatten(1).amax(1).clamp_min(1e-12).div(127.5).view(-1, *[1] * (w.dim() - 1))
    return w + ((w / s).round().clamp(-128, 127) * s - w).detach()


class FakeQuantS8(nn.Module):
    def forward(self, w):
        return fake_quant_s8(w)


def npu_layers(s: Student):
    """The conv and linear layers the NPU runs: all but the CPU heads and the branches of unfused RepConvs."""
    branches = {id(c) for r in s.modules() if isinstance(r, RepConv) and r.fused is None for c in (r.k3[0], r.k1[0])}
    return [m for n, m in s.named_modules()
            if isinstance(m, (nn.Conv2d, nn.Linear)) and not n.startswith("cpu_heads") and id(m) not in branches]


def quant_sim(s: Student, qat: bool = False) -> Student:
    """S8 weights on the NPU layers. Evaluation (a fused student): rounded in place. QAT (an unfused student): the
    RepConvs round their folded weights on every forward, the other layers through a parametrization, both with a
    straight-through gradient, so the RepVGG branches and BN affines train as before (the BN statistics stay fixed)."""
    from torch.nn.utils import parametrize
    if not qat:
        assert all(m.fused is not None for m in s.modules() if isinstance(m, RepConv)), "fuse() first"
        with torch.no_grad():
            for m in npu_layers(s):
                m.weight.copy_(fake_quant_s8(m.weight))
        return s
    if isinstance(s.backbone, R18Backbone):     # QAT rounds the BN-folded convs, as Pulsar2 does (BN statistics fixed)
        s.backbone.fuse_bn()
    for m in s.modules():
        if isinstance(m, RepConv):
            m.qat = True
    for m in npu_layers(s):
        parametrize.register_parametrization(m, "weight", FakeQuantS8())
    return s


def plain_state(s: Student) -> dict:
    """state_dict without the QAT parametrizations (the float weights under the plain names)."""
    return {k.replace("parametrizations.weight.original", "weight"): v for k, v in s.state_dict().items()}


def expand_state(s: Student, old: dict, old_arch: dict | None = None, keep=()) -> list[str]:
    """Load an unfused checkpoint's state into a student with a larger temporal input (policy_tokens, lane_policy) or
    temporal_heads, keeping its function: the new input columns start at zero. The temporal layer's history grows
    at the front (the oldest tokens come first) and the lane features come last; a temporal head's input gains the
    temporal features after the trunk. keep: key prefixes that keep the student's own values (e.g. a new pretrained
    backbone and its neck). Returns the expanded keys."""
    new = s.state_dict()
    grown = []
    front = (s.policy_tokens - (old_arch or {}).get("policy_tokens", 9)) * s.summarizer[-1].out_features
    for k, v in new.items():
        if k.startswith(tuple(keep)):
            continue
        w = old[k]
        if w.shape != v.shape:
            assert w.dim() == 2 and w.shape[0] == v.shape[0] and w.shape[1] < v.shape[1], (k, w.shape, v.shape)
            e = torch.zeros_like(v)
            if k == "temporal.0.weight":
                e[:, front:front + w.shape[1]] = w
            else:
                assert k.startswith("heads."), k
                e[:, :w.shape[1]] = w
            w = e
            grown.append(k)
        new[k] = w
    s.load_state_dict(new)
    return grown


def load_student(ck: dict) -> Student:
    """A checkpoint's student (eval mode, CPU); fused ones are built fused."""
    s = Student(**{**ck.get("arch", {}), "pretrained": False})
    if ck.get("fused"):
        s.fuse()
    elif ck.get("qat") and isinstance(s.backbone, R18Backbone):     # a QAT checkpoint holds the BN-folded convs
        s.backbone.fuse_bn()
    s.load_state_dict(ck["model"])
    return s.eval()


def count_macs(s: Student) -> dict:
    """Conv and linear MACs of one inference, by part."""
    macs = {}
    hooks = []

    def hook(name):
        def f(m, i, o):
            if isinstance(m, nn.Conv2d):
                n = o.numel() * m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]
            else:
                n = o.numel() * m.in_features
            macs[name] = macs.get(name, 0) + n
        return f

    for name, m in s.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)) and m.__class__ is not RepConv:
            part = name.split(".")[0]
            hooks.append(m.register_forward_hook(hook(part)))
    with torch.no_grad():
        s(torch.zeros(1, 12, 128, 256), torch.zeros(1, 12, 128, 256), torch.zeros(1, 25, 8), torch.zeros(1, 24, 512),
          torch.tensor([[1.0, 0.0]]))
    for h in hooks:
        h.remove()
    return macs
