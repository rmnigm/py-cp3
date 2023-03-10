# Энергия ферромагнетика

## **Описание модели**

### **Одномерный случай**

Модель Изинга является моделью магнетика. Пусть этот магнетик состоит из молекул, расположенных в узлах регулярной решетки. Пусть всего таких узлов будет $N$ штук, с индексами $i=1,\ldots, N$.

Предположим, что каждая молекула может быть представлена в виде магнитной стрелки, которая всегда либо направлена вдоль некоторой заданной оси, либо в противоположном направлении. То есть каждая молекула $i$ имеет две конфигурации, которые можно описывать с помощью "спиновой" переменной $\sigma_i$. Эта переменная принимает значение +1 (параллельно оси, спин направлен вверх) и -1 (антипараллельно оси, спин направлен вниз).

Пусть $\sigma = \{\sigma_1, \sigma_2, \ldots, \sigma_N\}$ обозначает набор значений всех $N$ спинов. Имеется $2^N$ различных наборов $\sigma$, и каждый из них описывает некоторое состояние системы.

Гамильтониан системы  состоит из двух частей: первая $E_0$ включает вклад межмолекулярных сил внутри магнетика, а вторая $E_1(\sigma)$ вклад от взаимодействий каждого спина с внешним магнитным полем (здесь считается нулевым). 

$$
H(\sigma)=E_0(\sigma)+E_1(\sigma)
$$

В любой физической системе мы предполагаем все взаимодействия инвариантными по отношению к обращению времени, что означает инвариантность $E$ при изменении знаков всех полей и намагниченностей. Энергия должна быть четной функцией от $\sigma$:

$$
E_0(\sigma_1,\ldots, \sigma_N)=E_0(-\sigma_1,\ldots, -\sigma_N)
$$

Энергия системы при нулевом внешнем магнитном поле равна сумме произведений **соседних** спинов на константы взаимодействия $J_{ij}$:

$$
E(\sigma) = -\sum_{i} J_{i,i+1}\sigma_{i}\sigma_{i+1} 
$$

Вероятность находиться в состоянии $\sigma$: 

$$
P(\sigma)=\frac{e^{-\beta E(\sigma)}}{Z},
$$

где $Z = \sum_{\sigma} e^{-\beta E(\sigma)}$ - статистическая сумма, $\beta = \frac{1}{k T}$ - обратная температура, $k$ -  константа Больцмана. Средняя энергия системы рассчитывается по всевозможным состояниям системы, т.е. всевозможным наборам $\sigma$:

$$
\langle E \rangle = \frac{1}{Z}\sum_{\{\sigma \}} E(\sigma)e^{-\frac{E(\sigma)}{kT}}
$$

### **Двумерный случай**

В случае двумерной решетки энергия системы при нулевом внешнем магнитном поле вычисляется следующим образом: 

$$
E(\sigma) = -\sum_{i,j} J_{ij}(\sigma_{i,j}\sigma_{i+1,j} + \sigma_{i,j}\sigma_{i,j+1})
$$


## Условия задачи

**Дано:**

- двумерная решетка молекул, расположенных в узлах кристаллической решетки, размеров $L_x \times L_y$ с периодическими границами
- каждая молекула обладает спином +1 или -1
- межмолекулярное взаимодействие описывается константами $J_{ij} = 1$
- модель Изинга

**Задача:**

- согласно модели Изинга рассчитать среднюю энергию $\langle E \rangle$ для указанной цепочки молекул при:
    - размерах решетки $L_x \in [2, 3, ..., 8], L_y = 4$
    - температурах $kT \in [1, 1.1, ..., 5.0]$
- сохранить массив средних энергий при помощи `np.save`
- вывести время расчета каждой итерации по $Lx / kT$.
- отобразить цветовую карту:
    - ось абсцисс - $L_x$
    - ось ординат - $kT$
    - цветом отобразить нормированное значение средней энергии $\frac{\langle E \rangle}{Lx Ly}$
    - подписать оси
    - отобразить цветовую шкалу (`colorbar`)
    - засечки должны соответствовать значениям $Lx, kT$
