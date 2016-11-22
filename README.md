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
# Feature : student, problem_hierarchy, problem_name, step_name, kc
python prob.py algebra_2005_2006
python prob.py algebra_2006_2007
python prob.py bridge_to_algebra_2006_2007
```
| Dataset      | Training    | Testing  |
| -------------|-------------|----------|
| algebra_2005_2006 | 0.0970676740172 | 0.427699903868 |
| algebra_2006_2007 | 0.0547102543474 | 0.403950930702 |
| bridge_to_algebra_2006_2007 | 0.0290279956492 | 0.371627041609 |

```sh
# Feature : student, problem_hierarchy, problem_name, step_name, kc, opportunity
```
| Dataset      | Training    | Testing  |
| -------------|-------------|----------|
| algebra_2005_2006 | 0.337000977608 | 0.42186015689 |
| algebra_2006_2007 | 0.335781933001 | 0.3990621855 |
| bridge_to_algebra_2006_2007 | 0.261519231749 | 0.357504131339 |

## SVM
```
python feature_vector.py algebra_2005_2006
```
| Dataset      | Training    | Testing  | Size|Running Time|
| -------------|-------------|----------|-----|------------|
| algebra_2005_2006 | 0.395655318254 | 0.463706747382 |50000|~30 min|
