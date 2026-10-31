---
name: user-working-style
description: How this user wants work reported and which safety rules they set for the supercombo_distill project
metadata:
  node_type: memory
  type: user
  originSessionId: fdabf76d-b147-4353-b3fb-ed51f739931a
  modified: 2026-10-30T12:02:46.387Z
---

The user writes short Korean instructions and wants answers in Korean: results as tables against the shipped model, plain status updates, no long recaps.

Standing rules they set:
- Passwords (Mac, MaixCAM2 board) come only from them and must never be written into files; pass them via environment variables (SSHPASS + SSH_ASKPASS helper on the Mac).
- Don't start training or servers without their instruction (a /goal they set counts as instruction for the goal's scope).
- Stop edgepilot.service on the MaixCAM2 before measuring and restart it afterwards.

They follow progress remotely, so finished artifacts (HANDOVER.md, reports/experiments.html) should be sent with SendUserFile when asked. See [[project-distill-status]].
