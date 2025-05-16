from numpy import random
import sys
numPoints = int(sys.argv[1])
typeDist = sys.argv[2]

dsx = 10000
dsy = 10000

x = []
y = []

print(numPoints, "2 0 0")


if typeDist == "zipf":
    print("Generating (x,y) points using a Zipf distribution", file = sys.stderr)
    x = random.zipf(1.1, numPoints)
    y = random.zipf(1.1, numPoints)
elif typeDist == "exp":
    print("Generating (x,y) points using a exponential distribution", file = sys.stderr)
    x = random.exponential(scale=1, size=numPoints)
    y = random.exponential(scale=1, size=numPoints)
else:
    print("Generating (x,y) points using a Normal distribution", file = sys.stderr)
    x = random.normal(loc=0.0, scale=dsx, size=numPoints)
    y = random.normal(loc=0.0, scale=dsy, size=numPoints)

for i in range(0,numPoints):
    print(i,x[i], y[i])

