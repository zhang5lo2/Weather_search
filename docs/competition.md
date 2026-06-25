# Competition Notes

- Official page: <https://www.raicom.com.cn/content.html?cid=1627>
- API title: `2026年智海算法调优赛题赛题`
- Downloaded rule file: `docs/raicom_2026_algorithm_tuning_rules.pdf`
- Local task shape: weather image classification with labels `cloudy`, `rainy`, `snowy`, and `sunny`.
- Evaluation focus: F1 score. Local training reports macro F1 on a stratified validation split.
- Inference target from FAQ: CPU, 2 cores, 8 GiB RAM.
- Platform torch version from FAQ: `2.1.7`. Public Conda/PyTorch packages expose the `2.1.x` line, so local environment files target PyTorch `2.1.x` and torchvision `0.16.x`.

The official page renders the detail dynamically through `https://service.raicom.com.cn/contentDetail?cid=1627`. The project keeps a local copy of the linked PDF so the code and environment can be reviewed without revisiting the page.
