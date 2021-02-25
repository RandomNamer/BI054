
from queue import PriorityQueue
from random import shuffle
import os

N=4
SEARCH_TYPE = "greedy" # implemented: greedy, A*, dijkstra

def moves(position):
    blank = position.index(N*N-1)
    i, j = divmod(blank, N)
    offsets = []
    if i>0: offsets.append(-N)  # Down
    if i<N-1: offsets.append(N) # Up
    if j>0: offsets.append(-1)  # Right
    if j<N-1: offsets.append(1) # Left
    for offset in offsets:
        swap = blank + offset
        yield tuple(position[swap] if x==blank else position[blank] if x==swap else position[x] for x in range(N*N))

def loss(position):
    return sum(abs(i//N - position[i]//N) + abs(i%N - position[i]%N) for i in range(N*N - 1))

def parity(permutation):
    #assert set(permutation) == set(range(N*N))
    #return sum(x<y and px>py for (x, px) in enumerate(permutation) for (y, py) in enumerate(permutation))%2
    seen, cycles = set(), 0
    for i in permutation:
        if i not in seen:
            cycles += 1
            while i not in seen:
                seen.add(i)
                i = permutation[i]
    return (cycles+len(permutation)) % 2

class Position: # For PriorityQueue, to make "<" do the right thing.
    def __init__(self, position, start_distance):
        self.position = position
        self.loss = loss(position)
        self.start_distance = start_distance
    def __lt__(self, other): # For A* and Dijkstra start_distance is indeed distance to start position
        if SEARCH_TYPE == "greedy":
            return self.loss < other.loss
        elif SEARCH_TYPE == "A*":
            return self.loss + self.start_distance < other.loss + other.start_distance
        elif SEARCH_TYPE == "dijkstra":
            return self.start_distance < other.start_distance
        else:
            raise NotImplementedError
    def __str__(self): return '\n'.join((N*'{:3}').format(*[(i+1)%(N*N) for i in self.position[i:]]) for i in range(0, N*N, N))

start = list(range(N*N-1))
while 1:
    shuffle(start)
    if parity(start) == 0: break
start += [N*N-1]
start = tuple(start)

p = Position(start, 0)
print ('Start shape:',p,p.position)
candidates = PriorityQueue()
candidates.put(p)
visited = set([p]) # Tuples rather than lists so they go into a set.
came_from = {p.position: None}

while p.position != tuple(range(N*N)):
    p = candidates.get()
    for k in moves(p.position):
        if k not in visited:
            candidates.put(Position(k,p.start_distance+1))
            came_from[k] = p
            visited.add(k)
res='Result: \n'
step=0
while p.position != start:
    res+='current move: \n'+str(p)+'\n'
    step+=1
    #print('current move: \n',p, "\n")
    p = came_from[p.position]
import resource
print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), 'RAM')
with open('res.txt','w+') as f:
    f.write(res)
    print(step)
    f.flush()
    f.close()
