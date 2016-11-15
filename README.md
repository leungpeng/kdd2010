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

# How to run
```sh
time python project.py algebra_2005_2006
time python project.py algebra_2006_2007
time python project.py bridge_to_algebra_2006_2007
```

# Result

## Version 1
Feature List:
```
studentId
problem_hierarchy
problem_step (problem_name + step_name) <- from http://pslcdatashop.org/KDDCup/workshop/papers/JMLR_Y10.pdf P.5
```

| Dataset      | Training    | Testing  |
| -------------|-------------|----------|
| algebra_2005_2006|0.0967046609696|0.410096769405|
| algebra_2006_2007|0.0547122669775|0.392632958868|
| bridge_to_algebra_2006_2007|0.0290291660401|0.349633911083|

## Version 2
