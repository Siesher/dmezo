---
description: Run Day 1 sanity check — MeZO on Qwen3-4B / SST-2
---

Запусти Day 1 sanity check:

1. Проверь, что torch видит GPU (`nvidia-smi` или `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`)
2. Запусти `python scripts/01_sanity_check_mezo.py --config configs/qwen3_4b_sst2.yaml`
3. Покажи финальные метрики: loss/accuracy на train и eval
4. Если accuracy < 50% (близко к случайному угадыванию на SST-2) — отметь это как red flag и предложи диагностику
