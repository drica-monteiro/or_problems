# Databricks notebook source
import numpy as np
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, LongType
import random

# COMMAND ----------

from pulp import PULP_CBC_CMD, HiGHS_CMD, LpVariable, LpBinary, LpInteger, LpProblem, LpMinimize, lpSum, value, HiGHS, LpStatus

# COMMAND ----------

import pyspark.sql.functions as f

# COMMAND ----------

values = [
    100, 95, 80, 90, 20, 15, 60, 55, 10, 50,
    33, 50, 61, 89, 81, 88, 100, 48, 25, 66,
    33, 42, 68, 85, 46, 21, 67, 86, 13, 58,
    34, 30, 42, 56, 31, 50, 99, 27, 57, 81
]

# COMMAND ----------

n = 40 #number of products
a = 50
b = 100
#stock = [np.random.randint(a,b) for p in range(0, n)] #how many units are available, ranging between 50 and 100
stock = values
prods = [f"p{i}" for i in range(0,n)] ##create n products
a = 0.2 
b = 20
weights = np.random.uniform(a, b, n)

# COMMAND ----------

data = [(p,g, float(round(w,2))) for p,g,w in zip(prods,stock, weights)]

# COMMAND ----------

schema = StructType([
    StructField("Product_id", StringType(), True),
    StructField("Stock", IntegerType(), True),
    StructField("Weights", FloatType(), True)
])

df_prods = spark.createDataFrame(data, schema=schema)

# COMMAND ----------

df_prods.show(5)

# COMMAND ----------

n = 81 ##number of slots
slots = list(np.arange(n)) 
corridor1 = [1 for c in range(27)]
corridor2 = [2 for c in range(27)]
corridor3 = [3 for c in range(27)]
#corridor4 = [4 for c in range(15)]
corridors = corridor1 + corridor2 + corridor3 #+ corridor4   
len(corridors)

# COMMAND ----------

capactity = [200 for p in range(n)]

# COMMAND ----------

nums = list(np.arange(1,32))
skip = [10, 11, 21, 22]
nums = [n for n in range(1, 32) if n not in skip]
distances = [n for n in nums for _ in range(3)]
len(distances)

# COMMAND ----------

nums1 = list(np.arange(4,7))
nums2 = list(np.arange(3,6))
nums3 = list(np.arange(1,4))
ergonomy = nums1*9 + nums2*9 + nums3*9

# COMMAND ----------

hei = list(np.arange(0,3))
height = hei*27
len(height)

# COMMAND ----------

data2 = [(s, cor, cap, dist, erg, hei) for s,cor,cap, dist, erg, hei in zip(slots,corridors,capactity,distances,ergonomy,height)]

# COMMAND ----------

schema2 = StructType([
    StructField("Slot", IntegerType(), True),
    StructField("Corridor", IntegerType(), True),
    StructField("Capacity", IntegerType(), True),
    StructField("Distance", IntegerType(), True),
    StructField("Ergonomy", IntegerType(), True),
    StructField("Height", IntegerType(), True)])



# COMMAND ----------

df_slots = spark.createDataFrame(data2, schema=schema2)
df_slots.show(5)

# COMMAND ----------

df_slots.groupBy('Corridor').count().show()

# COMMAND ----------

combo1 = [random.choice(prods) for p in range(8)]
combo2 = [random.choice(prods) for p in range(8)]
cols = ['p1', 'p2']
data = [(c1,c2) for c1,c2 in zip(combo1, combo2)]
df_combos = spark.createDataFrame(data, cols)
df_combos.show(5)

# COMMAND ----------

#####SOLVING THE PROBLEM

# COMMAND ----------

df_slots.show(5)

# COMMAND ----------

products = [r["Product_id"] for r in df_prods.select("Product_id").collect()]
slots    = [r["Slot"] for r in df_slots.select("Slot").collect()]

# COMMAND ----------

df_slots.show(5)

# COMMAND ----------

units = {r['Product_id']: float(r['Stock']) for r in df_prods.select('Product_id', 'Stock').collect()}
weight = {r['Product_id']: float(r['Weights']) for r in df_prods.select('Product_id', 'Weights').collect()}
distance = {r['Slot']: float(r['Distance']) for r in df_slots.select('Slot', 'Distance').collect()}
ergonomy = {r["Slot"]: float(r["Ergonomy"]) for r in df_slots.select('Slot', 'Ergonomy').collect()}
height = {r["Slot"]: float(r["Height"]) for r in df_slots.select('Slot', 'Height').collect()}
corridor = {r["Slot"]: float(r["Corridor"]) for r in df_slots.select('Slot', 'Corridor').collect()}
capacity = {r['Slot']: float(r['Capacity']) for r in df_slots.select('Slot', 'Capacity').collect()}


# COMMAND ----------

df_slots.groupBy('Corridor').count().show()

# COMMAND ----------

# Calculate neighbors
slots_by_corredor = (
    df_slots
    .groupBy("Corridor")
    .agg(f.collect_list("Slot").alias("slots"))
    .collect()
)

neighbors = {}
for row in slots_by_corredor:
    corredor_val = row["Corridor"]
    slots_corredor = sorted(row["slots"])
    for i in range(0, len(slots_corredor), 9): # each slot has 9 neighbors
        bloco = slots_corredor[i:i+9]
        for s in bloco:
            neighbors[s] = [v for v in bloco if v != s]

# COMMAND ----------

avaiable_corridors = sorted(set(corridor.values()))

# COMMAND ----------

# Model
model = LpProblem("Slotting", LpMinimize)

## variables
y = {(p, s): LpVariable(f"y_{p}_{s}", cat=LpBinary) for p in products for s in slots}
x = {(p, s): LpVariable(f"x_{p}_{s}", lowBound=0, cat=LpInteger) for p in products for s in slots}
z = {p: LpVariable(f"z_{p}", lowBound=0) for p in products}

# COMMAND ----------

## regularizing parameters for distance and ergonomy
theta = 10
mu = 10

model += (
    lpSum([y[p, s] * (distance[s] * units[p]) for p in products for s in slots]) +
    theta * lpSum([x[p, s] * weight[p] * ergonomy[s] for p in products for s in slots]) +
    mu * lpSum([x[p, s] * weight[p] * height[s] for p in products for s in slots])
)


# COMMAND ----------

print(f'We have {len(model.variables())} variables!') ## up to now, we have n_prods*n_slots*2 variables (the z variables is not considered yet)

# COMMAND ----------

# Constraints
for p in products:
    for s in slots:
        model += weight[p] * x[p, s] <= capacity[s] * y[p, s]
        model += y[p, s] <= x[p, s]

for p in products:
    model += lpSum([x[p, s] for s in slots]) == units[p]

for s in slots:
    model += lpSum([y[p, s] for p in products]) <= 1

for p in products:
    model += lpSum([y[p, s] for s in slots]) <= np.ceil(units[p] * weight[p] / capacity[slots[0]])

for p in products:
    for s1 in slots:
        for s2 in slots:
            if s1 >= s2:
                continue
            if s2 not in neighbors.get(s1, []):
                model += y[p, s1] + y[p, s2] <= 1


# COMMAND ----------

M = 3 #any big number
combo_produtos_df = df_combos.select("p1").union(
    df_combos.select("p2").withColumnRenamed("p2", "p1")
).distinct()

combo_produtos = [row["p1"] for row in combo_produtos_df.collect()]

for p in products:
    if p in combo_produtos:
        for s in slots:
            cs = corridor[s]
            model += z[p] - cs <= M * (1 - y[p, s])
            model += z[p] - cs >= -M * (1 - y[p, s])

for row in df_combos.select("p1", "p2").collect():
    p1 = row["p1"]
    p2 = row["p2"]
    model += z[p1] == z[p2]

# COMMAND ----------

##Now, the numbers of variables and constraints are
print("Number of variables:", len(model.variables()))
print("Number of constraints:", len(model.constraints))

# COMMAND ----------

solver_list = pulp.listSolvers(onlyAvailable=True)
print(solver_list)

# COMMAND ----------

# Solve
model.solve(HiGHS(msg=True))

# COMMAND ----------

print("Status:", LpStatus[model.status])