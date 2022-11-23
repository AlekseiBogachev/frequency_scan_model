# pydlts
## Краткое описание
### Назначение
Пакет модулей для обработки экспериментальных данных для моей кандидатской диссертации. Набор модулей реализует следующий функционал:
- чтение данных полученных с измерительной установки и их преобразование в единую таблицу (csv-файл);
- поиск оптимальных параметров модели на экспериментальных данных методом градиентного спуска;
- построение графиков с результатами разных этапов обработки;
- автоматическая генерация материалов для отчётов в LaTex;
- пакетная обработка результатов измерений и идентификация моделей.

Градиентный спуск реализован с помощью `TensorFlow`. 
Модели в `pydlts.fsmodels` сделаны совместимыми с `scikit-learn`.  
Нормализация данных сделана не совсем корректно с точки зрения математики
и требований структуры класса совместимого с scikit-learn.
Также в [отчёте](https://github.com/AlekseiBogachev/pydlts/blob/main/docs/report/models_main.pdf) раздел, посвящённый нормализации данных, написан с ошибкой.
Ошибки будут исправлены.

Чтобы поддерживать идентичность обработки разных комплектров результатов измерений, код оформлен ввиде пакета, который подключается с помощью `pip` и `setuptools`.

**Подробное описание работы модуля и решаемых задач можно найти в [отчёте о работе](https://github.com/AlekseiBogachev/pydlts/blob/main/docs/report/models_main.pdf).**

Документация на GitHub в процессе разработки.

### Типовой рабочий процесс
Обычно, обработка экспериментальных данных происходит в пакетном режиме и состоит из следующих шагов:
1. Пакетная обработка результатов измерений с помощью объекта `BatchDataReaderDLS82E` из модуля `pydlts.misc`:
    - сохранение csv-файлов с результатами измерений в формате, удобном для дальнейшей работы,
    - сохранение графиков, иллюстрирующих полученный результат.
1. Пакетная идентификация моделей с помощью объекта `BatchSingleExp` из модуля `pydlts.misc`:
    - сохранения csv-файлов с оптимальными найденными параметрами моделей,
    - генерация графиков, иллюстрирующих результаты.
1. Создание материалов для отчёта в LaTex на основе сохранённых результатов (пакетная обработка).

Также возможно обрабатывать результаты измерений индивидуально. В таком случае обработка будет состоять из следующих этапов:
1. Обработка результатов измерений (данных непосредственно с измерительной установки) с помощью объекта DataReaderDLS82E из модуля `pydlts.misc`.
1. Сохранение csv-файлов с результатами измерений в формате, удобном для дальнейшей работы, сохранение графиков, иллюстрирующих полученный результат.
Шаги 1 и 2 могут быть пропущены, если такая обработка уже выполнялась.
1. Идентификация параметров необходимой модели. Реализованы слеудющие 2:
    - моноэкспоненциальная модель с показателем нелинейности-неэкспоненциальности объект `SklSingleExpFrequencyScan` из модуля `pydlts.fsmodels`,
    - многоэкспоненциальня модель объект `SklMultiExpFrequencyScan` из модуля `pydlts.fsmodels`.
1. Создание необходимых графиков с помощью необходимых функций из `pydlts.fsplots`.
1. Создание отчёта вручную.

Пример результатов измерений, их обработки, идентификации моделей, графиков и отчёта представлен в репозитории [data-acquisition-and-report](https://github.com/AlekseiBogachev/data-acquisition-and-report).

## Описание структуры репозитория
Опущено описание файлов типовых для репозиториев на GitHub и файлов, необходимых для создания пакета.
Сырые данные с утановки не приводятся, можно посмотреть на их пример в [data-acquisition-and-report](https://github.com/AlekseiBogachev/data-acquisition-and-report).
- **datasets** - папка с результатами измерений
- **docs** - папка с документацией и отчётом.
    - **drafts** - черновики (будут удалены)
    - **report** - итоговый отчёт о результатах работы (LaTex). Кроеме проекта в Latex, папка также содержит вспомогательные ноутбуки, 
      нужные для некоторых расчётов и генерации иллюстраций.
        - **models_main.pdf** - отчёт в pdf
- **examples** - Примеры использования инструментов из пакета pydlts.
- **src** - Папка с исходными кодами.
    - **pydlts**
        - **fsmodels.py** - Модели экспериментальных данных.
        - **fsplots.py** - Функции для построения графиков.
        - **misc.py** - Вспомогательные классы и функции, реализующие преобразование сырых данных с измерительной установки в формат, 
          пригодный для работы, пакетную обработку, автоматическую генерацию материалов в отчёт.
- **tests** - Папка для автоматических тестов (пока остаётся пустой).
- **env.yml** - виртуальное окружение (windows), экспортированное из anaconda.
