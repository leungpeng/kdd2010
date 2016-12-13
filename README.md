# Installation
Python 2.7.11 or above

Package required: Scikit, matplotlib, numpy 

We recommand user to install Anaconda 4.2.0 or above

# Folder structure
```ls
.
README.md	dataset		project.py

./dataset/Development/algebra_2005_2006:
algebra_2005_2006.txt		algebra_2005_2006_master.txt	algebra_2005_2006_test.txt	algebra_2005_2006_train.txt

./dataset/Development/algebra_2006_2007:
algebra_2006_2007.txt		algebra_2006_2007_master.txt	algebra_2006_2007_test.txt	algebra_2006_2007_train.txt

./dataset/Development/bridge_to_algebra_2006_2007:
bridge_to_algebra_2006_2007.txt		bridge_to_algebra_2006_2007_master.txt	bridge_to_algebra_2006_2007_test.txt	bridge_to_algebra_2006_2007_train.txt

```


# Result

## All-in-one comparison

```sh
python all_in_one.py algebra_2005_2006
python all_in_one.py algebra_2006_2007
python all_in_one.py bridge_to_algebra_2006_2007
```
| bridge_to_algebra_2006_2007, NumberOfLineToTrain=100k, feature_vec=normal    | Test rmse   |
| ----------|-------------|
| KNN(k=5) | 0.4225|
| RandomForest(tree=50) | 0.5853|
| LinearSVM(C=1.0) | 0.7003|
| NaiveBayesian| 0.8423|
|Collabrative filtering(Rate=0.001, step=50, reg=0.02) | 0.3982 |

ROC curve:
![alt tag](https://raw.githubusercontent.com/leungpeng/cmsc5724_project/master/roc_bridge_to_0607_100kline.png?token=AQjFyqlXh219YLOj_cTbkU-_Tr9T9lONks5YWAR2wA%3D%3D)

| bridge_to_algebra_2006_2007, NumberOfLineToTrain=50k, feature_vec=normal    | Test rmse   |
| ----------|-------------|
| KNN(k=5) | 0.4408|
| RandomForest(tree=50) | 0.558|
|Collabrative filtering(Rate=0.01, step=50, reg=0.02) | 0.889 |

ROC curve:
![alt tag](https://raw.githubusercontent.com/leungpeng/cmsc5724_project/master/roc_bridge_to0607_50kline.png?token=AQjFyk5AuyJfWcbGlnQU0AIDTSCmTtSeks5YVpnLwA%3D%3D)


## Random method
```sh
python random_method.py algebra_2005_2006
python random_method.py algebra_2006_2007
python random_method.py bridge_to_algebra_2006_2007
```

| Dataset      | Training    | Testing  |
| -------------|-------------|----------|
| algebra_2005_2006 | 0.577339348465 | 0.578375289713 |
| algebra_2006_2007 | 0.577176770545 | 0.581733462971 |
| bridge_to_algebra_2006_2007 | 0.577114504595 | 0.578898326215 |

## Prob
```sh
# Feature : student, problem_hierarchy, problem_name, step_name, kc, opportunity
python prob.py algebra_2005_2006
python prob.py algebra_2006_2007
python prob.py bridge_to_algebra_2006_2007
```
| Dataset      | Training    | Testing  |Time|
| -------------|-------------|----------|----|
| algebra_2005_2006 | 0.0967263175545 | 0.426565287948 |0m52.609s|
| algebra_2006_2007 | 0.0541229228797 | 0.40384679442 |2m17.756s|
| bridge_to_algebra_2006_2007 | 0.0288237930397 | 0.372662133708 |11m34.139s|

```sh
# Feature : student, problem_hierarchy, problem_name, step_name, kc, opportunity
```
| Dataset      | Training    | Testing  |
| -------------|-------------|----------|
| algebra_2005_2006 | 0.233574258494 | 0.408946329676 |
| algebra_2006_2007 | 0.200404828212 | 0.39463380799 |
| bridge_to_algebra_2006_2007 | 0.172471892949 | 0.355436539753 |

## Collaborative Filtering
```sh
# Feature : student, problem_hierarchy, problem_name, step_name
python cf.py algebra_2005_2006
python cf.py algebra_2006_2007
python cf.py bridge_to_algebra_2006_2007
```
| Dataset      | Training    | Testing  |
| -------------|-------------|----------|
| algebra_2005_2006 | 0.415726052285 | 0.405040698634 |
| algebra_2006_2007 | 0.411450169266 | 0.4043896756 |
| bridge_to_algebra_2006_2007 | 0.311035090946 | 0.365743405094 |

## (1) Feature vector based method (OneHotEncoding)
```sh
# Feature : student(weight=50), unit, section, processed step_name, kc
python feature_vector.py algebra_2005_2006
```
| Dataset      | Method  | Training    | Testing  | Size|Training Time|Predict Time
| -------------|---------|-------------|----------|-----|------------|------------|
| algebra_2005_2006 | SVM |0.395655318254 | 0.463706747382 |50k|~30 min|     |
| algebra_2005_2006 | KNN(k=10) | 0.332238179089 | 0.517161982612 |50k| 9 sec| 17 sec  |
| algebra_2005_2006 | KNN(k=20) | 0.324500426044 | 0.46722690752 |200k| 147 sec| 558 sec  |
| algebra_2005_2006 | RandomForest(n=50) | 0.319 | 0.607 |50k| 5.7 sec| 0.8 sec  |

```sh
# Feature : student(weight=10), unit, section, process problem name, normalized problem view,
processed step_name, kc, normalized opportunity
python feature_vector.py algebra_2005_2006
```
| Dataset      | Method  | Training    | Testing  | Size|Training Time|Predict Time
| -------------|---------|-------------|----------|-----|------------|------------|  |
| algebra_2005_2006 | KNN(k=40) | 0.139213460584 | 0.481050023681 |50k| 15 sec| 73 sec  |
| algebra_2005_2006 | KNN(k=200) | 0.137313853181 | 0.451867815084 |200k| 175 sec| 1774 sec  |


```sh
# Feature : student(weight=10), unit, section, process problem name, normalized problem view,
processed step_name, kc, normalized opportunity, 8 cfars, 3 temporal (<6 mean cfars,
<6 mean hints, whether has this <student,KC> pairs)
python feature_vector.py algebra_2005_2006
```
| Dataset      | Method  | Training    | Testing  | Size|Training Time|Predict Time
| -------------|---------|-------------|----------|-----|------------|------------|  |
| algebra_2005_2006 | KNN(k=50) | 0.12016775277 | 0.480263350998 |50k| 12 sec| 124 sec |

```sh
# Feature : student(weight=10), unit, section, process problem name, normalized problem view,
processed step_name, kc, normalized opportunity, 8 cfars, 3 temporal (<6 mean cfars, <6 mean hints,
whether has this <student,KC> pairs), 4 time memory {same day, one week, one month, >one month}
of same person and KC tuple
python feature_vector.py algebra_2005_2006
```
| Dataset      | Method  | self test(first50k row)    | Testing(rows ID<max trained rows) | Size|Training Time|Predict Time
| -------------|---------|-------------|----------|-----|------------|------------|  |
| algebra_2005_2006 | KNN(k=10) | 0.0835463942968 | 0.489339023162 |200k| 268 sec| 854 sec |
| algebra_2005_2006 | KNN(k=1000) | 0.0516397779494 | 0.43467239637 |100k| 19 sec| 40 sec |

## (2) Feature vector based method (unique key for each column, distance between vectors are not defined)
```
python feature_vector.py algebra_2005_2006
```
| Dataset      | Method  | Training    | Testing  | Size|Training Time|Predict Time
| -------------|---------|-------------|----------|-----|------------|------------|
| algebra_2005_2006 | RandomForest(n=50) |  0.282914828894 | 0.562234728644 |100k| 1.3 sec| 0.4 sec  |
| algebra_2005_2006 | MultinomialNB |  0.61850935768 | 0.563801780499 |100k| 0.09 sec| 0.05 sec  |
| algebra_2005_2006 | KNN(n=500) | 0.292019565059 | 0.46641692094 |200k| 5 sec| 21 sec  |
| algebra_2005_2006 | KNN(n=1000) | 0.291 | 0.462 |all rows| 146 sec| 1180 sec  |
