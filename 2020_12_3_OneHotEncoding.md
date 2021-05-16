# One Hot Encoding

단순 수치형 데이터가 아닌 문자형 데이터를 만나게 되는 경우에 사용
지역을 선택하고 이에 관한 데이터를 분석할 때, 사람은 지역에 해당하는 번호를 부여할 때에는 번호의 큰 의미를 두지 않지만, 컴퓨터는 사람과 달라 서로 서열이 있거나 거리가 먼 것으로 인식

  이럴 경우 데이터를 처리하는 것은 어려워지게 됨으로 One Hot Encoding을 사용하여  
  범주형 변수가 가지는 값 만큼 변수를 만들고 자기 위치만 1, 나머지는 0을 만들어서 변환 하는 것을 말한다.


```python
import pandas as pd

df = pd.read_csv('insurance.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>sex</th>
      <th>bmi</th>
      <th>children</th>
      <th>smoker</th>
      <th>region</th>
      <th>charges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>female</td>
      <td>27.900</td>
      <td>0</td>
      <td>yes</td>
      <td>southwest</td>
      <td>16884.92400</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>male</td>
      <td>33.770</td>
      <td>1</td>
      <td>no</td>
      <td>southeast</td>
      <td>1725.55230</td>
    </tr>
    <tr>
      <th>2</th>
      <td>28</td>
      <td>male</td>
      <td>33.000</td>
      <td>3</td>
      <td>no</td>
      <td>southeast</td>
      <td>4449.46200</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>male</td>
      <td>22.705</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>21984.47061</td>
    </tr>
    <tr>
      <th>4</th>
      <td>32</td>
      <td>male</td>
      <td>28.880</td>
      <td>0</td>
      <td>no</td>
      <td>northwest</td>
      <td>3866.85520</td>
    </tr>
  </tbody>
</table>
</div>



### region에 One Hot Encoding 적용하기 

범주형 변수가 가지는 값 만큼 변수를 만들고 자기 위치만 1, 나머지는 0을 만들어서 변환
region의 경우는

* 'northeast': [1, 0, 0, 0]
* 'northwest': [0, 1, 0, 0]
* 'southeast': [0, 0, 1, 0]
* 'southwest': [0, 0, 0, 1]
로 변환


```python
set(df['region'].to_numpy()) 
```




    {'northeast', 'northwest', 'southeast', 'southwest'}



* ``.to_numpy()) ``란 데이터프레임형태로 되어있는 값들을 array형태로 바꾸는 함수 


```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')

region = df['region'].to_numpy().reshape(-1, 1)

enc.fit(region)
print(enc.categories_)
print(region[:5])
print(enc.transform(region[:5]).toarray())
```

    [array(['northeast', 'northwest', 'southeast', 'southwest'], dtype=object)]
    [['southwest']
     ['southeast']
     ['southeast']
     ['northwest']
     ['northwest']]
    [[0. 0. 0. 1.]
     [0. 0. 1. 0.]
     [0. 0. 1. 0.]
     [0. 1. 0. 0.]
     [0. 1. 0. 0.]]
    

### DataFrame에 있는 범주형 변수 여러 개를 한꺼번에 처리하기 


```python
# 범주형 변수들 불러오기
df_category = df.iloc[:,[1,4,5]]
df_category.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sex</th>
      <th>smoker</th>
      <th>region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>female</td>
      <td>yes</td>
      <td>southwest</td>
    </tr>
    <tr>
      <th>1</th>
      <td>male</td>
      <td>no</td>
      <td>southeast</td>
    </tr>
    <tr>
      <th>2</th>
      <td>male</td>
      <td>no</td>
      <td>southeast</td>
    </tr>
    <tr>
      <th>3</th>
      <td>male</td>
      <td>no</td>
      <td>northwest</td>
    </tr>
    <tr>
      <th>4</th>
      <td>male</td>
      <td>no</td>
      <td>northwest</td>
    </tr>
  </tbody>
</table>
</div>




```python
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(df_category)
X_category = enc.transform(df_category).toarray()
X_category[:5]
```




    array([[1., 0., 0., 1., 0., 0., 0., 1.],
           [0., 1., 1., 0., 0., 0., 1., 0.],
           [0., 1., 1., 0., 0., 0., 1., 0.],
           [0., 1., 1., 0., 0., 1., 0., 0.],
           [0., 1., 1., 0., 0., 1., 0., 0.]])


