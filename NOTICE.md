# Notices

This repository contains research code and documentation for pi0 fine-tuning, data collection, and real-robot evaluation.

## Third-party Pika SDK

`collect/pika_sdk/` is a vendored third-party SDK provided by Songling/Pika. It is included only so the data-collection code can call the Pika Sense and Pika gripper interfaces.

The project MIT license does not relicense `collect/pika_sdk/`. Use, modification, and redistribution of that SDK remain subject to the terms from its original provider.

## Large Artifacts

Datasets, model checkpoints, Hugging Face caches, training outputs, and local archive files are intentionally excluded from the public source tree. Publish those artifacts separately when needed.
