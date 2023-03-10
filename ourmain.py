import json
import math
import numpy as np
from tqdm import tqdm
import time
from statistics import mean

with open('Cost.json') as json_file:
    cost = json.load(json_file)

with open('Dist.json') as json_file:
    Dist = json.load(json_file)

with open('G.json') as json_file:
    G = json.load(json_file)

with open('Coord.json') as coord_file:
    coord = json.load(coord_file)

class Graph(object):

    def __init__(self, vertices_no):
        self.vertices_no = vertices_no
        self.graph = {}
    
    def add_edge(self, start, destination, weight, cost):
        exists = self.graph.get(start)
        if exists:
            self.graph[start][destination] = [weight, cost]
        else:
            self.graph[start] = { destination: [weight, cost] }

    def getGraph(self):
        return self.graph

    def print_graph(self):
        for key, values in self.graph.items():
            print('Adjacent of ', key, ': ', values)

class minHeap:

    def __init__(self, maxsize):
        self.size = 0
        self.maxsize = maxsize
        self.Heap = [[0,0]] * (self.maxsize + 1)
        self.Heap[0][1] = -1 * math.inf
        self.FRONT = 1
        #self.Heap[0][1] = -1 * inf #Initialise root element to minimum

    def getSize(self):
        return self.size

    def parent(self, pos):
        return pos//2
    
    def leftchild(self, pos):
        return pos * 2
    
    def rightchild(self, pos):
        return (pos * 2) + 1
    
    def isLeaf(self, pos):
        return pos * 2 > self.size

    def swap(self, pos1, pos2):
        self.Heap[pos1], self.Heap[pos2] = self.Heap[pos2], self.Heap[pos1]
    
    def heapify(self, pos):
            if not self.isLeaf(pos):
                if (self.Heap[pos][1] > self.Heap[self.leftchild(pos)][1] or
                self.Heap[pos][1] > self.Heap[self.rightchild(pos)][1]):
                    if self.Heap[self.leftchild(pos)][1] < self.Heap[self.rightchild(pos)][1]:
                        self.swap(pos, self.leftchild(pos))
                        self.heapify(self.leftchild(pos))

                    else:
                        self.swap(pos, self.rightchild(pos))
                        self.heapify(self.rightchild(pos))

    def insert(self, element, weight):
        if self.size >= self.maxsize:
            return
        insertitem = [element, weight]

        self.size += 1
        self.Heap[self.size] = insertitem
        current = self.size 
        
        while self.Heap[current][1] < self.Heap[self.parent(current)][1]:
            self.swap(current, self.parent(current))
            current = self.parent(current)


    def totalheapify(self):
        for pos in range(self.size // 2, 0, -1):
            self.heapify(pos)


    def extractCheapest(self):
        if self.size <= 0:
            return
        popped = self.Heap[self.FRONT]
        self.Heap[self.FRONT] = self.Heap[self.size]
        self.size -= 1
        self.heapify(self.FRONT)
        return popped

class priorityQueueHeap:

    def __init__(self, maxsize):
        self.heap = minHeap(maxsize)
        self.size = 0
        
    def isEmpty(self):
        if self.heap.getSize() == 0:
            return True

    def delete(self):
        self.size -= 0
        return self.heap.extractCheapest()

    def insert(self, element, weight):
        self.heap.insert(element, weight)
        self.size += 1 

    def extractCheapest(self):
        self.size -= 1
        return self.heap.extractCheapest()

numberOfNodes = len(G)
graph = Graph(numberOfNodes)

print('=' * 30, ' Generating Graph ', '=' * 30)
for key in tqdm(Dist):
    distance = Dist[key]
    splitted = key.split(',')
    sourceNode = int(splitted[0])
    destinationNode = int(splitted[1])
    edge_cost = cost.get(key, math.inf)
    graph.add_edge(sourceNode, destinationNode, distance, edge_cost)

print()
print('=' * 30, '      Task 1     ', '=' * 30)

def ucs(g, src, destination):
    visited = {  }
    queue = priorityQueueHeap(len(g))
    distance = {}
    accumulated_cost = {}
    for i in range(1, len(g) + 1):
        distance[i] = math.inf
        visited[i] = 0
        accumulated_cost[i] = 0
    distance[src] = 0
    queue.insert(src, 0)
    predecessor = { src : -1 }

    while not queue.isEmpty():
        node = queue.extractCheapest()
        current = node[0]

        visited[current] = 1
        if current == destination:
            return predecessor, distance, accumulated_cost, visited

        for node, info in g[current].items():
            edge_distance = info[0]
            edge_cost = info[1]
            if visited[node] == 0 and distance[node] > distance[current] + edge_distance:
                distance[node] = distance[current] + edge_distance
                predecessor[node] = current
                accumulated_cost[node] = accumulated_cost[current] + edge_cost
                queue.insert(node, distance[node])
   
def print_route(predecessor, source, destination):
    route = []
    while destination != 1:
        route.append(destination)
        destination = predecessor[destination]

    route = [str(index) for index in route]
    route.append(str(source))
    route.reverse()

    output_string = '->'.join(route)
    return output_string

timelist = []
for i in tqdm(range(1,21)):
    start = time.time()
    g = graph.getGraph()
    predecessor, distance, accumulated_cost, visited = ucs(g, 1, 50)
    end = time.time()
    timelist.append(end - start)

task1_average_time = mean(timelist)
print('Shortest path: ', print_route(predecessor, 1, 50))
print('Shortest distance: ', distance[50])
print('Total energy cost: ', accumulated_cost[50])
print('Average time taken: ', task1_average_time)
print('Number of nodes visited: ', sum(visited.values()))
print()



print('=' * 30, '      Task2      ', '=' * 30)
def ucs2(g, src, destination):

    visited = {}
    queue = priorityQueueHeap(len(g))
    distance = {}
    accumulated_cost = {}
    for i in range(1, len(g) + 1):
        distance[i] = math.inf
        visited[i] = 0
        accumulated_cost[i] = 0
    distance[src] = 0
    queue.insert(src, 0)
    predecessor = { src : 1 }

    while not queue.isEmpty():
        node = queue.extractCheapest()
        current = node[0]
        visited[current] = 1
        if current == destination:
            return predecessor, distance, accumulated_cost, visited

        for node, info in g[current].items():
            edge_distance = info[0]
            edge_cost = info[1]
            if accumulated_cost[current] + edge_cost > 287932:
                continue
            if visited[node] == 0 and distance[node] > distance[current] + edge_distance:
                distance[node] = distance[current] + edge_distance
                accumulated_cost[node] = accumulated_cost[current] + edge_cost
                predecessor[node] = current
                queue.insert(node, distance[node])

timelist = []
for i in tqdm(range(1,21)):
    start = time.time()
    g = graph.getGraph()
    predecessor, distance, accumulated_cost, visited = ucs2(g, 1, 50)
    end = time.time()
    timelist.append(end - start)

task1_average_time = mean(timelist)
print('Shortest path: ', print_route(predecessor, 1, 50))
print('Shortest distance: ', distance[50])
print('Total energy cost: ', accumulated_cost[50])
print('Average time taken: ', task1_average_time)
print('Number of nodes visited: ', sum(visited.values()))
print()



print('=' * 30, '      Task3      ', '=' * 30)
heuristics = {}
for key, item in coord.items():
    x1 = item[0]
    y1 = item[1]
    destx = coord['50'][0]
    desty = coord['50'][1]
    hn = math.sqrt((destx - x1)**2 + (desty - y1)**2) #+ abs(destx - x1 + desty - y1)
    heuristics[int(key)] = hn


def astar(g, src, destination):

    visited = {}
    queue = priorityQueueHeap(len(g))
    distance = {}
    accumulated_cost = {}
    for i in range(1, len(g) + 1):
        distance[i] = math.inf
        visited[i] = 0
        accumulated_cost[i] = 0
    distance[src] = 0
    queue.insert(src, heuristics[1])
    predecessor = { src : 1 }

    while not queue.isEmpty():
        node = queue.extractCheapest()
        current = node[0]
        visited[current] = 1
        if current == destination:
            return predecessor, distance, accumulated_cost, visited

        for node, info in g[current].items():
            edge_distance = info[0]
            edge_cost = info[1]
            if accumulated_cost[current] + edge_cost > 287932:
                continue
            if visited[node] == 0 and distance[node] > distance[current] + edge_distance:
                distance[node] = distance[current] + edge_distance
                accumulated_cost[node] = accumulated_cost[current] + edge_cost
                predecessor[node] = current
                queue.insert(node, distance[node] + heuristics[node])

timelist = []
for i in tqdm(range(1,21)):
    start = time.time()
    g = graph.getGraph()
    predecessor, distance, accumulated_cost, visited = astar(g, 1, 50)
    end = time.time()
    timelist.append(end - start)

task1_average_time = mean(timelist)
print('Shortest path: ', print_route(predecessor, 1, 50))
print('Shortest distance: ', distance[50])
print('Total energy cost: ', accumulated_cost[50])
print('Average time taken: ', task1_average_time)
print('Number of nodes visited: ', sum(visited.values()))



