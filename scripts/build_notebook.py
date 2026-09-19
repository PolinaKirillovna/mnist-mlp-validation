"""Build ``notebooks/lab1_mnist_validation.ipynb``.

The notebook is a thin, fast walk-through: it imports the package, does a few
light live calls (loading, statistics, metrics on the test set, one occlusion
map) and displays the tables and figures produced by the full pipeline
(``make validate grid interpret jax``). It contains no heavy re-computation, so
it executes in well under a minute. Text cells (in Russian) carry the analysis.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path("notebooks/lab1_mnist_validation.ipynb")


def md(text: str) -> nbf.NotebookNode:
    """Return a markdown cell."""
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    """Return a code cell."""
    return nbf.v4.new_code_cell(text)


CELLS = [
    md(
        "# ЛР1. Валидация данных, обучение и интерпретация MLP на MNIST\n\n"
        "Ноутбук — сквозной проход по всем частям работы через вызовы пакета "
        "`mnist_validation`. Тяжёлые вычисления (полная сетка обучения, окклюзия "
        "по всем изображениям, сравнение фреймворков) выполняются командами "
        "`make validate grid interpret jax`; здесь мы делаем лёгкие живые вызовы "
        "и показываем готовые таблицы и рисунки из `reports/`."
    ),
    code(
        "import json\n"
        "from pathlib import Path\n\n"
        "import pandas as pd\n"
        "from IPython.display import Image, display\n\n"
        "from mnist_validation.config import load_config\n"
        "from mnist_validation.data.loading import load_bundle_npz\n\n"
        "ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
        "FIG = ROOT / 'reports' / 'figures'\n"
        "TAB = ROOT / 'reports' / 'tables'\n"
        "config = load_config(ROOT / 'configs' / 'lab1_base.yaml')\n"
        "print('seed', config.seed, '| hidden', config.model.hidden_sizes)"
    ),
    md(
        "## Часть 1. Анализ и валидация набора\n\nЖивой расчёт статистик классов и "
        "репрезентативности на обработанных данных, затем — готовые таблицы и рисунки."
    ),
    code(
        "from mnist_validation.data.statistics import class_distribution_frame, imbalance_ratio\n\n"
        "test = load_bundle_npz(ROOT / 'data' / 'processed' / 'test.npz')\n"
        "original = load_bundle_npz(ROOT / 'data' / 'processed' / 'original.npz')\n"
        "print('коэффициент дисбаланса (обучающая часть):', round(imbalance_ratio(original.labels), 3))\n"
        "display(class_distribution_frame(original.labels))"
    ),
    code(
        "display(pd.read_csv(TAB / 'part1_quality_summary.csv'))\n"
        "display(pd.read_csv(TAB / 'part1_representativeness.csv'))"
    ),
    code(
        "for name in ['part1_class_histogram', 'part1_examples', 'part1_umap',\n"
        "             'part1_anomalies', 'part1_dataset_sizes']:\n"
        "    display(Image(str(FIG / f'{name}.png')))"
    ),
    md(
        "Набор количественно репрезентативен (объёма хватает всем классам), но сильно "
        "несбалансирован (класс 3 раздут дубликатами, класс 9 урезан). Аномалии "
        "(~2,35 %) сохранены для Части 3."
    ),
    md(
        "## Часть 2. Обучение и сравнение моделей\n\nСводка сетки экспериментов и "
        "итоговые метрики трёх моделей."
    ),
    code(
        "grid = pd.read_csv(TAB / 'part2_grid.csv')\n"
        "display(grid.sort_values('val_macro_f1', ascending=False).head(10))\n"
        "display(pd.read_csv(TAB / 'part2_final_metrics.csv'))\n"
        "print(json.loads((TAB / 'part2_summary.json').read_text())['best_by_dataset'])"
    ),
    code(
        "for name in ['part2_grid_curves', 'part2_hyperparams', 'part2_metric_bars',\n"
        "             'part2_confusion_original']:\n"
        "    display(Image(str(FIG / f'{name}.png')))"
    ),
    md(
        "Лучшая модель — на исходном наборе (больше данных перевешивает дисбаланс); "
        "на меньших сбалансированном/очищенном наборах оптимален dropout 0,5."
    ),
    md(
        "## Часть 3. Оценка и интерпретация\n\nЖивой расчёт метрик и одной карты "
        "окклюзии на загруженной итоговой модели, затем готовые рисунки/таблицы."
    ),
    code(
        "import numpy as np\n"
        "from mnist_validation.models.torch_mlp import load_torch_model, resolve_device\n"
        "from mnist_validation.evaluation.metrics import compute_metrics\n"
        "from mnist_validation.preprocessing import to_float01\n\n"
        "device = resolve_device('cpu')\n"
        "clf = load_torch_model(ROOT / 'artifacts' / 'final' / 'original.pt', device)\n"
        "metrics = compute_metrics(test.labels, clf.predict_logits(to_float01(test.images).astype('float64')))\n"
        "print('test accuracy:', round(metrics.accuracy, 4), '| macro-F1:', round(metrics.macro_f1, 4))"
    ),
    code(
        "from mnist_validation.interpretation.occlusion import occlusion_map\n\n"
        "pos = int(np.flatnonzero(test.labels == 3)[0])\n"
        "image = to_float01(test.images[pos:pos+1]).astype('float64')[0]\n"
        "res = occlusion_map(clf, image, true_label=3, kernel=3)\n"
        "print('предсказанный класс:', res.predicted_class, '| базовая вероятность:', round(res.base_prob_pred, 3))\n"
        "print('форма карты важности:', res.importance_pred.shape)"
    ),
    code(
        "display(pd.read_csv(TAB / 'part3_confused_pairs.csv').head(5))\n"
        "for name in ['part3_disagreement', 'part3_occlusion_class3', 'part3_weights_original',\n"
        "             'part3_activation_original']:\n"
        "    display(Image(str(FIG / f'{name}.png')))"
    ),
    md(
        "Аномальные объекты в большинстве по-прежнему уверенно и верно "
        "классифицируются (в среднем 2,74 из 3 моделей согласны с исходной меткой), "
        "поэтому их удаление почти не влияет на метрики."
    ),
    md("## Часть 4. Сравнение PyTorch и JAX и градиентные методы"),
    code(
        "display(pd.read_csv(TAB / 'part4_frameworks.csv'))\n"
        "display(pd.read_csv(TAB / 'part4_error_overlap.csv'))\n"
        "display(pd.read_csv(TAB / 'part4_map_correlation.csv'))"
    ),
    code(
        "for name in ['part4_framework_bars', 'part4_stability', 'part4_curves_original',\n"
        "             'part4_gradients_class3']:\n"
        "    p = FIG / f'{name}.png'\n"
        "    if p.exists():\n"
        "        display(Image(str(p)))"
    ),
    md(
        "PyTorch и JAX при одинаковых условиях дают близкие метрики и сильно "
        "пересекающиеся множества ошибок; расхождения объясняются разной "
        "инициализацией, реализацией dropout и численной точностью. Градиентные "
        "карты (Saliency, Input×Gradient, Integrated Gradients) согласуются между "
        "фреймворками и коррелируют с картами окклюзии."
    ),
    md(
        "## Выводы\n\nПолные выводы — в `reports/report.md`. Ключевое: набор "
        "количественно репрезентативен, но несбалансирован и содержит дубликаты и "
        "выбросы; двухслойный перцептрон достигает ~0,97–0,98 macro-F1; интерпретация "
        "(окклюзия, карты весов, градиентные методы) показывает, что модель опирается "
        "на осмысленные штрихи цифр; PyTorch и JAX сопоставимы."
    ),
]


def main() -> None:
    """Assemble and write the notebook."""
    notebook = nbf.v4.new_notebook()
    notebook.cells = CELLS
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NB_PATH)
    print(f"Wrote {NB_PATH} with {len(CELLS)} cells")


if __name__ == "__main__":
    main()
