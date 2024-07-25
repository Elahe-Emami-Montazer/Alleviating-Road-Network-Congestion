# ya HAGH

import numpy as np
import pydtmc
import pylab as p
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt

import MCTA


def generate_graph(nodes, edges):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges, weight='weight')

    return G


def show_graph(G):
    pos = nx.circular_layout(G)
    nx.draw(G, pos, with_labels=True, connectionstyle='arc3, rad = 0.1')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    plt.show()
    plt.savefig("graph.png")


def compute_shortest_paths(G, src_dst):
    paths = []
    for i in src_dst:
        paths.append(nx.dijkstra_path(G, source=i['src'], target=i['dst'], weight='weight'))

    # print(paths)
    return paths


def compute_p(G, paths):
    nodes = list(G.nodes)
    edges = list(G.edges)

    # Count the number of cars exiting each intersection
    p = np.zeros(shape=(len(nodes), len(nodes)))
    for path in paths:
        for k in range(len(path) - 1):
            p[path[k]][path[k + 1]] += 1

    # print(p)
    # each row should sum up to 1
    i = 0
    for t in p:
        row_sum = np.sum(t)
        if row_sum != 0:
            for j in range(len(t)):
                p[i][j] /= row_sum

        i += 1

    return p


def convert_to_paths_edge_based(paths):
    # convert paths from (node to node) to (edge to edge)
    paths_edge_based = []
    for path in paths:
        path_edge = [(path[i], path[i + 1]) for i in range(0, len(path) - 1, 1)]
        paths_edge_based.append(path_edge)

    return paths_edge_based


def compute_p_dual(G, paths):
    nodes = list(G.nodes)
    edges = list(G.edges)

    # convert paths from (node to node) to (edge to edge)
    paths_edge_based = convert_to_paths_edge_based(paths)

    # print(paths_edge_based)

    num_edges = len(edges)
    # print(num_edges)
    p_dual = np.zeros(shape=(num_edges, num_edges))

    # print(edges)
    for i, src_edge in enumerate(edges):
        for j, dst_edge in enumerate(edges):
            for path in paths_edge_based:
                if src_edge in path:
                    if dst_edge in path:
                        if (path.index(src_edge) + 1 == path.index(dst_edge)):
                            p_dual[i][j] += 1

    # print(p_dual)

    all_zero_row = np.where(~p_dual.any(axis=1))[0]
    for i in range(len(all_zero_row)):
        p_dual[all_zero_row[i]][0] = 1

    # print(p_dual)
    # each row should sum up to 1
    i = 0
    for t in p_dual:
        row_sum = np.sum(t)
        if row_sum != 0:
            for j in range(len(t)):
                p_dual[i][j] /= row_sum

        i += 1

    #print(p_dual)
    return p_dual


def compute_normalized_tt(lengths, speed):
    # edges = G.edges(data='weight')
    tt = []
    for length in lengths:
        tt.append(length / speed)

    for t in range(len(tt)):
        tt[t] /= min(tt)

    # print(tt)
    return tt


def compute_modified_tpm(tpm, tt):
    n = len(tt)
    modified_tpm = np.zeros(shape=(n, n))
    # print(tpm)
    # print(n)
    for i in range(n):
        modified_tpm[i][i] = ((tt[i] - 1) / tt[i])

    for i in range(n):
        for j in range(n):
            if i != j:
                modified_tpm[i][j] = (1 - modified_tpm[i][i]) * tpm[i][j]

    # print(modified_tpm)
    return modified_tpm


def near(a, b, rtol=1e-5, atol=1e-8):
    return np.abs(a - b) < (atol + rtol * np.abs(b))


def steady_state_prob(p):
    # values, vectors = sp.sparse.linalg.eigs(p, k=1, sigma=1)
    values, vectors = sp.linalg.eig(p, left=True, right=False)
    # print(values)
    vectors = vectors.T
    vector = vectors[near(values, 1)]

    steady_state = []
    if len(vector) == 0:
        print('--- no steady state ---')
        exit(0)
    else:
        state = (vector / np.sum(vector))[0]

        for i, s in enumerate(state):
            steady_state.append(np.round(state[i].real, 6))

    return steady_state


def density(num_cars, stedy_state, road_len, num_lines):
    density = []
    n = len(stedy_state)
    for i in range(n):
        density.append((num_cars * stedy_state[i]) / (num_lines * road_len[i]))

    return density


def compute_cost_of_each_edge(length, density):
    cost = []
    normalized_density = []
    max_density = max(density)
    max_len = max(length)
    for i in range(len(density)):
        normalized_density.append((density[i] / max_density) * max_len)

    # print(normalized_density)

    for i in range(len(density)):
        cost.append(length[i] + 100 * density[i])

    max_cost = max(cost)
    # normalize the cost
    # for i, c in enumerate(cost):
    #     cost[i] /= max_cost

    # print(f'cost: {cost}')
    return cost


def find_cost_of_edge(edge_costs, edge):
    ret = None
    for e, c in edge_costs:
        if e == edge:
            ret = c
            break

    return ret


def compute_cost_of_each_path(edges, costs, paths):
    edge_costs = list(zip(edges, costs))
    costs = []

    paths_edge_based = convert_to_paths_edge_based(paths)

    for path in paths_edge_based:
        cost = 0
        for edge in path:
            cost += find_cost_of_edge(edge_costs, edge)

        costs.append(cost)

    # print(costs)
    return costs


def main_program(G, src_dst, road_len, speed, num_lines, num_cars):
    # compute shortest paths for each src and dst
    paths = compute_shortest_paths(G, src_dst)
    # print(paths)

    # compute tpm of the primal matrix
    p = compute_p(G, paths)
    # print("transition probability matrix:")
    # print(p)

    # compute tpm of the dual matrix (edge to edge)
    tpm = compute_p_dual(G, paths)
    # print(G.edges)
    # print("tpm of the dual matrix:")
    # print(tpm)

    # compute travel times of each edge
    tt = compute_normalized_tt(road_len, speed)
    # print(tt)

    modified_tpm = compute_modified_tpm(tpm, tt)
    # print("modified_tpm:")
    # print(modified_tpm)

    steady_state = steady_state_prob(modified_tpm)
    # print("steady_state:")
    # print(steady_state)

    dens = density(num_cars, steady_state, road_len, num_lines)
    # print("density")
    # print(dens)
    # to plot the density
    # colors = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # plt.scatter(range(len(dens)) ,dens, c=colors, cmap='viridis')
    # plt.show()

    edge_cost = compute_cost_of_each_edge(road_len, dens)
    # print("cost of each edge:")
    # print(edge_cost)
    # to plot the edge costs
    # colors = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # plt.scatter(range(len(edge_cost)) ,edge_cost, c=colors, cmap='viridis')
    # plt.show()

    path_costs = compute_cost_of_each_path(G.edges, edge_cost, paths)
    # to plot the path costs
    colors = range(len(path_costs))
    plt.scatter(range(len(path_costs)), path_costs, c=colors, cmap='viridis')
    plt.xlabel('paths (for each vehicle)')
    plt.ylabel('cost')
    plt.show()

    avg_paths_cost = sum(path_costs) / len(path_costs)
    # print(f'sum of cost of shortest path: {sum(path_costs)}')
    print(f'average cost of shortest path: {avg_paths_cost}')

    return edge_cost


if __name__ == '__main__':
    # ---------------------------- simulation inputs ---------------------------- #
    # the graph of roads
    nodes = [0, 1, 2, 3, 4, 5]

    edges = [(0, 1, 8), (0, 3, 4),
             (1, 2, 10), (1, 4, 2),
             (2, 0, 1), (2, 3, 3),
             (3, 5, 5),
             (4, 1, 2), (4, 5, 8),
             (5, 0, 4), (5, 2, 4)
             ]

    # speed of cars in each road: 60 km/h
    speed = 60

    # number of lines of each road
    num_lines = 1

    # traffic flows (sources and destinations)
    # [src, dst] -> chosen randomly
    src_dst = [{'src': 0, 'dst': 1},
               {'src': 0, 'dst': 2},
               {'src': 0, 'dst': 2},
               {'src': 0, 'dst': 3},
               {'src': 0, 'dst': 3},
               {'src': 0, 'dst': 3},
               {'src': 0, 'dst': 3},
               {'src': 0, 'dst': 4},
               {'src': 0, 'dst': 4},
               {'src': 0, 'dst': 5},
               {'src': 0, 'dst': 5},
               {'src': 1, 'dst': 0},
               {'src': 1, 'dst': 0},
               {'src': 1, 'dst': 2},
               {'src': 1, 'dst': 3},
               {'src': 1, 'dst': 4},
               {'src': 1, 'dst': 5},
               {'src': 1, 'dst': 5},
               {'src': 2, 'dst': 0},
               {'src': 2, 'dst': 0},
               {'src': 2, 'dst': 1},
               {'src': 2, 'dst': 3},
               {'src': 2, 'dst': 4},
               {'src': 2, 'dst': 5},
               {'src': 3, 'dst': 0},
               {'src': 3, 'dst': 1},
               {'src': 3, 'dst': 4},
               {'src': 3, 'dst': 5},
               {'src': 3, 'dst': 5},
               {'src': 4, 'dst': 0},
               {'src': 4, 'dst': 0},
               {'src': 4, 'dst': 0},
               {'src': 4, 'dst': 0},
               {'src': 4, 'dst': 1},
               {'src': 4, 'dst': 2},
               {'src': 4, 'dst': 3},
               {'src': 4, 'dst': 5},
               {'src': 5, 'dst': 0},
               {'src': 5, 'dst': 1},
               {'src': 5, 'dst': 1},
               {'src': 5, 'dst': 2},
               {'src': 5, 'dst': 2},
               {'src': 5, 'dst': 3},
               {'src': 5, 'dst': 3},
               {'src': 5, 'dst': 3},
               {'src': 5, 'dst': 4}
               ]

    num_cars = len(src_dst)
    road_len = []
    for r in edges:
        # weights of edges are road distances at the beginning
        road_len.append(r[2])

    G = generate_graph(nodes, edges)

    roads = list(zip(G.edges, road_len))
    # print(roads)

    # --------------------------------------------------------------------------- #

    # print(G.edges.data())
    # show_graph(G)

    edge_costs = main_program(G, src_dst, road_len, speed, num_lines, num_cars)
    # print(edge_costs)

    while True:
        avg_costs = sum(edge_costs) / len(edge_costs)
        updated_G = G.copy()
        # update the graph with new costs
        i = 0
        for s, d, w in updated_G.edges(data=True):
            w['weight'] = edge_costs[i]
            i += 1

        edge_costs = main_program(updated_G, src_dst, road_len, speed, num_lines, num_cars)
        new_avg_costs = sum(edge_costs) / len(edge_costs)
        # print(f'avg: {avg_costs}')
        # print(f'new: {new_avg_costs}')
        if np.isclose(avg_costs, new_avg_costs):
            break
