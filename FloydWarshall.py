def transform(adjacency):
    INF = 100000
    for i in range(0, len(adjacency)):

        for j in range(0, len(adjacency[0])):

            if adjacency[i][j] == 0 and i != j:
                adjacency[i][j] = INF

    return adjacency


def floydWarshall(adjacency):
    graph = transform(adjacency)

    V = len(graph[0])

    dist = graph.copy()

    for k in range(V):

        # pick all vertices as source one by one

        for i in range(V):

            # Pick all vertices as destination for the

            # above picked source

            for j in range(V):
                # If vertex k is on the shortest path from

                # i to j, then update the value of dist[i][j]

                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    return dist

