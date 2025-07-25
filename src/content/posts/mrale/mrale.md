---
title: mRALE Mastermind Challenge
published: 2023-08-15
description: 'MIDRC mRALE Mastermind Challenge; AI to predict severity on chest radiographs'
image: './mrale.png'
tags: [covid, chest x-ray]
category: 'works'
draft: false 
---

[Link to Challenge Site](https://www.midrc.org/mrale-mastermind-2023)

[Link to Challenge Review](https://www.youtube.com/watch?v=uuR4q38Qhdo)

The mRALE Mastermind Challenge was a competition organized by MIDRC — the Medical Imaging and Data Resource Center — a collaborative initiative supported by the National Institute of Biomedical Imaging and Bioengineering (NIBIB), aiming to foster innovation and open access to high-quality medical imaging data. The challenge tasked participants with building AI models to predict mRALE scores from chest radiographs — a scoring system used by radiologists to quantify the severity of lung involvement in COVID-19 infections.

The goal was to advance the development of machine learning methods that can assist clinicians in assessing disease progression and severity more efficiently and consistently. The dataset comprised labeled chest X-rays from a diverse patient population, and submissions were judged based on how accurately they could predict expert-assigned mRALE scores.

We are proud to share that our team placed 6th overall in the challenge using an ensemble method. Rather than relying on a single model, we combined predictions from multiple convolutional neural networks (CNNs) — including variants of ResNet, DenseNet, and EfficientNet — to improve generalization and reduce variance. Each model was trained with different preprocessing pipelines and data augmentations to encourage diversity in learned representations. The ensemble strategy allowed us to balance the strengths of each architecture and achieve a robust final prediction.

Participating in the mRALE Mastermind Challenge was an exciting opportunity to apply deep learning to a meaningful clinical problem, and it was inspiring to see the community come together to push the boundaries of AI in medical imaging.

# Contributors
Cohen Archbold, Atik Ahamed, Imran Abdullah-Al-Zubaer