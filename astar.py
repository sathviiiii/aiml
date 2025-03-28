graph_nodes = {'A': [('B', 1), ('C', 4)],
    'B': [('C', 2), ('D', 5)],
    'C': [('D', 1)],
    'D': []}
def get_neighbour(v):
    if v in graph_nodes:
        return graph_nodes[v]
    return None

def h(n):
    h_dist = {'A': 6,
        'B': 4,
        'C': 2,
        'D': 0}
    return h_dist[n]

def astar(start,end):
    opens = set(start)
    close = set()
    g = {}
    parents = {}
    g[start] = 0
    parents[start] = start

    while(len(opens)>0):
        n=None
        for v in opens:
            if n==None or g[v]+h[v] < g[n]+h[n]:
                n=v
        if n==end or graph_nodes[n]==None:
            pass
        else:
            for (m,wt) in get_neighbour(n):
                if m not in opens and m not in close:
                    opens.add(m)
                    g[m] = g[n]+wt
                    parents[m] = n
                else:
                    if(g[m]>g[n]+wt):
                        g[m] = g[n]+wt
                        parents[m] = n
                        if m in close:
                            close.remove(m)
                            opens.add(m)
            
            if n==None:
                print("No path")
                return None
            if n==end:
                path = []
                while parents[n]!=n:
                    path.append(n)
                    n = parents[n]
                path.append(start)
                path.reverse()
                print("Path found: {}".format(path))

        opens.remove(n)
        close.add(n)
    print("No path")
    return None

astar('A', 'D')