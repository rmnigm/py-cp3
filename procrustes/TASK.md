# Прокрустов анализ
### Условия задачи
**Дано:**

- 5 датасетов: `tiny`, `small`, `medium`, `large`, `xlarge`
- каждый датасет содержит массив фигур (кривых) в трехмерном пространстве, а каждая фигура состоит из 1000 точек
- три основных алгоритма для анализа датасетов и распределения на кластеры:
  - [прокрустово преобразование](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)
  - метод главных компонент [`PCA`](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
  - метод [кластеризации](https://scikit-learn.org/stable/modules/clustering.html), например, [`k-средних`](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.k_means.html)

**Задача:**

- выполнить описанные ниже требования для каждого датасета:
- для каждой пары фигур выполнить прокрустово преобразование и вычислить расстояние между кривыми
    - расстояние между кривыми = среднее евклидово расстояние между соответствующими точками фигур (т.е. точками с одинаковыми индексами)
- во время выполнения расчетов выводить время, затраченное на них
- построить матрицу расстояний между кривыми, где в i-й строке (и столбце) расположен вектор расстояний от i-й кривой до всех остальных
- отобразить матрицу расстояний на рисунке
- к матрице расстояний применить метод главных компонент (`PCA`):
    - выделить 2 главных направления (компоненты)
    - спроецировать векторы расстояний на плоскость, заданную найденными направлениями
    - на выходе будут получена матрица из двумерных векторов (`проекция`)
- полученную `проекцию` отобразить на рисунке
- определить количество кластеров (визуально или автоматически методом)
- применить метод кластеризации к векторам `проекции`:
    - определить номер кластера для каждого вектора `проекции` (для каждой фигуры)
    - определить центры кластеров
- отобразить `проекцию`, центры и номера кластеров
- для каждого кластера построить рисунок:
    - три проекции фигур этого кластера (`x-y`, `x-z`, `y-z`)


**Материалы:**
- [Прокрустов анализ](https://en.wikipedia.org/wiki/Procrustes_analysis)
- [Прокрустово преобразование](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)
- [Метод главных компонент](https://ru.wikipedia.org/wiki/Метод_главных_компонент)
- [Методы кластеризации](https://scikit-learn.org/stable/modules/clustering.html)
- [Метод k-средних](https://ru.wikipedia.org/wiki/Метод_k-средних)