# 알고리즘 글쓰기

## 2차원 List

- 1차원 List를 묶어놓은 List
- 2차원 이상의 다차원 List는 차원에 따라 Index를 선언
- 2차원 List의 선언: 세로길이(row 개수), 가로길이(col 개수) 를 필요로 함

```python
# 2행 4열의 2차원 List 
arr = [[0,1,2,3], 
				[4,5,6,7]]

# Shape 
| 0 | 1 | 2 | 3 |
| 4 | 5 | 6 | 7 |
```

## List 초기화 (ft. Python)

**(방법1)** 직접 나열 

**(방법2)** 반복되는 원소는 곱셈연산을 통해 표현; (시퀀스 형태의 자료형은 전부 가능함)

**(방법3)** List Comprehension

```python
# === 1차원 리스트 초기화 === # 

arr = [0,1,2,3,4,5]
arr = [0] * 5   # [0,0,0,0,0]
arr = [i for i in range(2,9) if i%2==0]   #[2,4,6,8]
```

```python
# === 2차원 리스트 초기화 === # 
brr = [[1,2,3], [1,2,3], [1,2,3]]
brr = [[1,2,3]]*3    # [[1,2,3], [1,2,3], [1,2,3]]
brr = [[1,2,3] for i in range(3)] # [[1,2,3], [1,2,3], [1,2,3]]
brr = [[i,j] for i in range(3) for j in range(2)] 
											# [[0,0],[0,1],[1,0],[1,1],[2,0],[2,1]]
```

## 2차원 List 입력 받기

```python
# === 입력 받을 데이터 === #  
3 4     # data shape 
0 1 0 0 # data 
0 0 0 0 
0 0 1 0 
```

- 첫째 줄에 $n$행 $m$열
- 둘째 줄부터 $n\times m$ 의 행렬 데이터가 주어질 경우 입력을 받는 방법

**(ex1)** 행 축별로 입력 

```python
# === 데이터 형태 입력 === # 
n, m = map(int, input().split()) 

# === 데이터 입력 === # 
data = [0 for _ in range(n)] # 행 개수 
														 # [0] * n 

for i in range(n): # 행 별로 값 입력
    data[i] = list(map(int, input().split()))
```

**(ex2)** `.append()` 메소드 사용

```python
# === 데이터 형태 입력 === # 
n, m = map(int, input().split()) 

# === 데이터 입력 === # 
data = [] 

for i in range(n): 
    data.append(list(map(int, input().split())))
```

**(ex3)** List comprehension 활용 

```python
# === 데이터 형태 입력 === # 
n, m = map(int, input().split()) 

# === 데이터 입력 === # 
data = [list(map(int, input().split())) for _ in range(n)]
```

## 2차원 List에서 원하는 데이터의 위치 찾기

```python
# === 입력 받을 데이터 === #  
3 4     # data shape 
0 1 0 0 # data 
0 0 0 0 
0 0 1 0 
```

- 1 이 입력된 (행, 열) 위치 찾기

**(sol)**

```python
# === 데이터 형태 입력 === # 
n, m = map(int, input().split()) 

# === 데이터 입력 === # 
data = [list(map(int, input().split())) for _ in range(n)]

# === 위치 찾기 === # 
location = [(i, j) for i in range(n) for j in range(m) if data[i][j]==1]
```

## 2차원 List 순회 (Iteration)

“$n \times m$ List의 $n*m$ 개의 모든 원소를 빠짐없이 조사하는 방법”

**[종류]**

- 행 우선 순회
- 열 우선 순회
- 지그재그(zig-zag) 순회

**행 우선 순회** 

“List의 행을 우선으로 List의 원소를 조사하는 방법” 

```python
[----------------->]
[----------------->]
[----------------->]
```

```python
arr=[[0,1,2,3],
		 [4,5,6,7],
		 [8,9,10,11]]

for i in range(len(arr)):  # 행 개수 
    for j in range(len(arr[i])): 
        print(f"{arr[i][j]}")
```

**열 우선 순회** 

“List의 열부터 먼저 조사하는 방법”

```python
|   |   |   |
|   |   |   |
|   |   |   |
v   v   v   v
```

```python
arr=[[0,1,2,3],
		 [4,5,6,7],
		 [8,9,10,11]]

for j in range(len(arr[0])): # 열 개수
    for i in range(len(arr)): 
        print(f"{arr[i][j]}")
```

**지그재그 순회**

“List의 행을 좌우로 조사하는 방법” 

```python
[----------------->]   # 순방향 
[<-----------------]   # 역방향 
[----------------->]   # 순방향 
```

```python
arr=[[0,1,2,3],
		 [4,5,6,7],
		 [8,9,10,11]]

n = len(arr)    # 행 길이 
m = len(arr[0]) # 열 길이

for i in range(len(arr)):
    for j in range(len(arr[0])): 
        item = arr[i][j+(m-2*j-1)*(i%2)]
        print(f"{item}")
```

- 순방향 ⇒ `arr[i][j]`
- 역방향 ⇒ `arr[i][ j + (m-2*j-1) ]`
- selector ⇒ `(i%2)`

※ 인덱스 값을 내림차순으로 나타내는 함수 (짝수 번째 행에서, $y=1$):

idx = $-j+m-1$ 

$m=4$ 일 때 

- $j=0$  ⇒ idx=3
- $j=1$  ⇒ idx=2
- $j=2$  ⇒ idx=1
- $j=3$  ⇒ idx=0

※ 인덱스 값을 오름차순으로 나타내는 함수 (홀수 번째 행에서, $y=0$):

idx = $j$ 

- $j=0$  ⇒ idx=0
- $j=1$  ⇒ idx=1
- $j=2$  ⇒ idx=2
- $j=3$  ⇒ idx=3

※ 위의 두 수식을 하나로 표현 하면? 

idx = $j + (m-2j-1)y$ 

- $y=0$  ⇒  idx = $j$
- $y=1$  ⇒  idx = $m-j-1$

## 델타($\Delta$)를 이용한 2차 List 탐색

“2차 List의 한 좌표에서 (상하좌우)**네 방향의 인접 List 요소를 탐색**할 때 사용하는 방법

- 델타 값은 한 좌표에서 네 방향의 좌표와 $x, y$ 의 **차이를 저장한 List로 구현**
- 델타 값을 이용하여 **특정 원소의 상하좌우에 위치**한 원소에 접근할 수 있음

**(Tip)** 이차원 List의 가장자리(edge) 원소들은 상하좌우 네 방향에 원소가 존재하지 않을 경우가 있으므로, **Index를 체크하거나 Index의 범위를 제한**해야 함

**EXAMPLE**: 델타($\Delta$)를 이용한 2차 List 탐색 

```python
arr = [ [1,2,3,4],
        [5,6,7,8],
        [9,10,11,12]]

dy = [-1,1,0,0] # 상하
dx = [0,0,-1,1] # 좌우 

# === 탐색 시작 === # 
for y in range(len(arr)):
    for x in range(len(arr[y])):
        print(f"neighbors of {arr[y][x]} @ point ({y},{x})")

        for i in range(4): # 상하좌우 
            X = x + dx[i]
            Y = y + dy[i]
            
            if (Y<0 or X<0) or (Y>=len(arr) or X>=len(arr[y])): # 범위제한 
                continue

            print(f"{arr[Y][X]} @ ({Y},{X})")
```

```python
# === output === # 
neighbors of 1 @ point (0,0)
5 @ (1,0)
2 @ (0,1)
neighbors of 2 @ point (0,1)
6 @ (1,1)
1 @ (0,0)
3 @ (0,2)
neighbors of 3 @ point (0,2)
7 @ (1,2)
2 @ (0,1)
4 @ (0,3)
neighbors of 4 @ point (0,3)
8 @ (1,3)
3 @ (0,2)
neighbors of 5 @ point (1,0)
1 @ (0,0)
9 @ (2,0)
6 @ (1,1)
neighbors of 6 @ point (1,1)
2 @ (0,1)
10 @ (2,1)
5 @ (1,0)
7 @ (1,2)
neighbors of 7 @ point (1,2)
3 @ (0,2)
11 @ (2,2)
6 @ (1,1)
8 @ (1,3)
neighbors of 8 @ point (1,3)
4 @ (0,3)
12 @ (2,3)
7 @ (1,2)
neighbors of 9 @ point (2,0)
5 @ (1,0)
10 @ (2,1)
neighbors of 10 @ point (2,1)
6 @ (1,1)
9 @ (2,0)
11 @ (2,2)
neighbors of 11 @ point (2,2)
7 @ (1,2)
10 @ (2,1)
12 @ (2,3)
neighbors of 12 @ point (2,3)
8 @ (1,3)
11 @ (2,2)
```

## 전치 행렬 (Transpose)

“행과 열의 값이 반대인 행렬을 의미” 

`arr[i][j]` ⇒ `arr[j][i]`

```python
arr = [ [1,2,3],
        [4,5,6],
        [7,8,9],]

for i in range(len(arr)): # 행 좌표
    for j in range(len(arr[i])): # 열 좌표
        
        if i < j : 
            arr[i][j], arr[j][i] = arr[j][i], arr[i][j] # swap           

print(f"{arr}")
```

```python
# === output === # 
[[1, 4, 7], [2, 5, 8], [3, 6, 9]]
```

**(Tip)** 모든 좌표에 대해 행과 영의 값을 바꾸면 **본래의 모습으로 되돌아오기 때문에 주의**하기

- 즉, $i < j$  조건을 붙여서 이 문제를 해결함

`**zip(iterable*)`**  

“**동일한 개수**로 이루어진 자료형들을 **묶어 주는 역할**을 하는 함수” 

![Untitled](%E1%84%8B%E1%85%A1%E1%86%AF%E1%84%80%E1%85%A9%E1%84%85%E1%85%B5%E1%84%8C%E1%85%B3%E1%86%B7%20%E1%84%80%E1%85%B3%E1%86%AF%E1%84%8A%E1%85%B3%E1%84%80%E1%85%B5%20840f5f2ceeb544a59d95bf4a86ccd6bb/Untitled.png)

- 동일한 길이의 리스트
- `zip()` 함수에 입력  ⇒ 각 리스트를 슬라이스(slice)하여 Tuple 객체로 묶어줌

**(ex1)** 기본 예시

```python
alpha = ['a', 'b', 'c']
index = [1, 2, 3]

alpha_index = list(zip(alpha, index))
print(f"{alpha_index}")

# === output === # 
[('a', 1), ('b', 2), ('c', 3)]
```

**(ex2)** 행별로 묶기

```python
arr = [[1,2,3], 
        [4,5,6], 
        [7,8,9]]

result = list(zip(*arr))

print(f"{result}")

# === output === # 
[(1, 4, 7), (2, 5, 8), (3, 6, 9)]
```

- 위의 방법을 통해 전치 행렬(transpose matrix)을 쉽게 구할 수 있음
- 즉, `zip(*matrix)` ⇒  전치 행렬

**(덧)** (ex2)를 시각화하면 

```python
zip(*arr) 는 아래와 같음: 
zip( [1,2,3], [4,5,6], [7,8,9]) 

즉, zip( [1,2,3], 
				[4,5,6], 
				[7,8,9])  

# == Tuple로 묶어서 반환 == # 
(1,4,7), (2,5,8), (3,6,9)
```

---

**Summary** 

- 2차원 List **선언** 및 **초기화** 방법
- 2차원 List **입력** 받기
- List indexing
- 2차원 List 순회(iteration)
- 델타($\Delta$) 값을 이용한 이웃 요소 탐색
- 전치 행렬 만드는 방법

---

**Reference** 

[1] [https://swexpertacademy.com/main/main.do](https://swexpertacademy.com/main/main.do)