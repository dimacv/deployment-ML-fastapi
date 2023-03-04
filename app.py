# импортируем библиотеки numpy и pandas
import numpy as np
import pandas as pd
# импортируем модуль cloudpickle
import cloudpickle
# Мы импортируем FastAPIкласс Python, обеспечивающий все функции API.
from fastapi import FastAPI
# из фреймворка Pydantic импортируем класс BaseModel,
# он будет нужен для проверки корректности входных данных
from pydantic import BaseModel
# импортируем Request объект класса из FastAPI,
# чтобы затем передать его в качестве параметра нашей basic_predict
from fastapi import Request

###########################################################################################################
###########################################################################################################
#with open('cloudpickle_for_deployment.pkl', mode='wb') as file:
#    cloudpickle.dump(pipe, file)

# загрузим сохраненную ранее модель
with open('cloudpickle_for_deployment.pkl', mode='rb') as file:
   pipe = cloudpickle.load(file)


# функция предварительной подготовки
def preprocessing(df):

    # значения переменной age меньше 18 заменяем
    # минимально допустимым значением возраста
    df['age'] = np.where(df['age'] < 18, 18, df['age'])

    # создаем переменную Ratio - отношение количества
    # просрочек 90+ к общему количеству просрочек
    sum_of_delinq = (df['NumberOfTimes90DaysLate'] +
                     df['NumberOfTime30-59DaysPastDueNotWorse'] +
                     df['NumberOfTime60-89DaysPastDueNotWorse'])

    cond = (df['NumberOfTimes90DaysLate'] == 0) | (sum_of_delinq == 0)
    df['Ratio'] = np.where(
        cond, 0, df['NumberOfTimes90DaysLate'] / sum_of_delinq)

    # создаем индикатор нулевых значений переменной
    # NumberOfOpenCreditLinesAndLoans
    df['NumberOfOpenCreditLinesAndLoans_is_0'] = np.where(
        df['NumberOfOpenCreditLinesAndLoans'] == 0, 'T', 'F')

    # создаем индикатор нулевых значений переменной
    # NumberRealEstateLoansOrLines
    df['NumberRealEstateLoansOrLines_is_0'] = np.where(
        df['NumberRealEstateLoansOrLines'] == 0, 'T', 'F')

    # создаем индикатор нулевых значений переменной
    # RevolvingUtilizationOfUnsecuredLines
    df['RevolvingUtilizationOfUnsecuredLines_is_0'] = np.where(
        df['RevolvingUtilizationOfUnsecuredLines'] == 0, 'T', 'F')

    # преобразовываем переменные в категориальные, применив
    # биннинг и перевод в единый строковый формат
    for col in ['NumberOfTime30-59DaysPastDueNotWorse',
                'NumberOfTime60-89DaysPastDueNotWorse',
                'NumberOfTimes90DaysLate']:
        df.loc[df[col] > 3, col] = 4
        df[col] = df[col].apply(lambda x: f"cat_{x}")

    # создаем список списков - список 2-факторных взаимодействий
    lst = [
        ['NumberOfDependents',
         'NumberOfTime30-59DaysPastDueNotWorse'],
        ['NumberOfTime60-89DaysPastDueNotWorse',
         'NumberOfTimes90DaysLate'],
        ['NumberOfTime30-59DaysPastDueNotWorse',
         'NumberOfTime60-89DaysPastDueNotWorse'],
        ['NumberRealEstateLoansOrLines_is_0',
         'NumberOfTimes90DaysLate'],
        ['NumberOfOpenCreditLinesAndLoans_is_0',
         'NumberOfTimes90DaysLate']
    ]

    # создаем взаимодействия
    for i in lst:
        f1 = i[0]
        f2 = i[1]
        df[f1 + ' + ' + f2 + '_interact'] = (df[f1].astype(str) + ' + '
                                             + df[f2].astype(str))

    # укрупняем редкие категории
    interact_columns = df.columns[df.columns.str.contains('interact')]
    for col in interact_columns:
        df.loc[df[col].value_counts()[df[col]].values < 55, col] = 'Other'

    return df

#################################################################################################

# создаем экземпляр класса FastAPI и называем его app
app = FastAPI()

@app.get("/")
async def root():
    return {"Message": "This is a test path for a general FastAPI health check."}

# Определение пути для прогнозирования без проверки данных
@app.post('/basic_predict')
async def basic_predict(request: Request):

    # Получение JSON из тела запроса
    input_data = await request.json()

    # Преобразование JSON в Pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # выполняем предварительную обработку новых данных
    input_df = preprocessing(input_df)

    # вычисляем вероятности для новых данных
    output = pipe.predict_proba(input_df)[:, 1][0]

    # возвращаем вывод пользователю
    return output

'''
# Для проверки этого пути в bash консоли можно набрать такой POST запрос:
curl -X 'POST' \
   'http://127.0.0.1:8000/basic_predict' \
   -H 'accept: application/json' \
   -d '{"RevolvingUtilizationOfUnsecuredLines": 0.88551908, "age": 43.0, 
   "NumberOfTime30-59DaysPastDueNotWorse": 0.0, "DebtRatio": 0.177512717, 
   "MonthlyIncome": 5700.0, "NumberOfOpenCreditLinesAndLoans": 4.0, 
   "NumberOfTimes90DaysLate": 0.0, "NumberRealEstateLoansOrLines": 0.0, 
   "NumberOfTime60-89DaysPastDueNotWorse": 0.0, "NumberOfDependents": 0.0}'
   
# output: 0.48133564201075385
'''

#################################################################################################


##########################################################################################
class InputData(BaseModel):

    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int


# Определение пути для прогнозирования с проверкой данных
@app.post('/predict')
async def predict(data: InputData):
    # Преобразование входных данных в Pandas DataFrame
    input_df = pd.DataFrame([data.dict()])

    # осуществляем замену символов в названиях столбцов входного датафрейма
    # для соответствия входным данным ожидаемых моделью
    input_df.columns = input_df.columns.str.replace('_', '-')

    # выполняем предварительную обработку новых данных
    input_df = preprocessing(input_df)

    # вычисляем вероятности для новых данных
    output = pipe.predict_proba(input_df)[:, 1][0]

    # возвращаемое значение
    return output

'''
# Для проверки этого пути в bash консоли можно набрать такой POST запрос:

curl -X 'POST' 'http://127.0.0.1:8000/predict' -H 'Content-Type: application/json' \
  -d '{"RevolvingUtilizationOfUnsecuredLines": 0.5, "age": 40,
  "NumberOfTime30_59DaysPastDueNotWorse": 2, "DebtRatio": 0.9,
  "MonthlyIncome": 7000, "NumberOfOpenCreditLinesAndLoans": 2,
  "NumberOfTimes90DaysLate": 2, "NumberRealEstateLoansOrLines": 2,
  "NumberOfTime60_89DaysPastDueNotWorse": 2, "NumberOfDependents": 2 }'

# outpot: 0.6146162030411374
'''