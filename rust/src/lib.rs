// Implements the dijkstra algorithm with cutoff, and multiple sources from networkx in rust.
// No need predecesors, paths, or target for now.

// class Graph:
//     def __init__(self, incoming_graph_data=None, **attr):
//         """Initialize a graph with edges, name, or graph attributes.
//
//         Parameters
//         ----------
//         incoming_graph_data : input graph (optional, default: None)
//             Data to initialize graph. If None (default) an empty
//             graph is created.  The data can be an edge list, or any
//             NetworkX graph object.  If the corresponding optional Python
//             packages are installed the data can also be a 2D NumPy array, a
//             SciPy sparse array, or a PyGraphviz graph.
//
//         attr : keyword arguments, optional (default= no attributes)
//             Attributes to add to graph as key=value pairs.
//
//         See Also
//         --------
//         convert
//
//         Examples
//         --------
//         >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
//         >>> G = nx.Graph(name="my graph")
//         >>> e = [(1, 2), (2, 3), (3, 4)]  # list of edges
//         >>> G = nx.Graph(e)
//
//         Arbitrary graph attribute pairs (key=value) may be assigned
//
//         >>> G = nx.Graph(e, day="Friday")
//         >>> G.graph
//         {'day': 'Friday'}
//
//         """
//         self.graph = self.graph_attr_dict_factory()  # dictionary for graph attributes
//         self._node = self.node_dict_factory()  # empty node attribute dict
//         self._adj = self.adjlist_outer_dict_factory()  # empty adjacency dict
//         # attempt to load graph with data
//         if incoming_graph_data is not None:
//             convert.to_networkx_graph(incoming_graph_data, create_using=self)
//         # load graph attributes (must be after convert)
//         self.graph.update(attr)
//
//     @cached_property
//     def adj(self):
//         """Graph adjacency object holding the neighbors of each node.
//
//         This object is a read-only dict-like structure with node keys
//         and neighbor-dict values.  The neighbor-dict is keyed by neighbor
//         to the edge-data-dict.  So `G.adj[3][2]['color'] = 'blue'` sets
//         the color of the edge `(3, 2)` to `"blue"`.
//
//         Iterating over G.adj behaves like a dict. Useful idioms include
//         `for nbr, datadict in G.adj[n].items():`.
//
//         The neighbor information is also provided by subscripting the graph.
//         So `for nbr, foovalue in G[node].data('foo', default=1):` works.
//
//         For directed graphs, `G.adj` holds outgoing (successor) info.
//         """
//         return AdjacencyView(self._adj)
//
//     def add_edges_from(self, ebunch_to_add, **attr):
//         """Add all the edges in ebunch_to_add.
//
//         Parameters
//         ----------
//         ebunch_to_add : container of edges
//             Each edge given in the container will be added to the
//             graph. The edges must be given as 2-tuples (u, v) or
//             3-tuples (u, v, d) where d is a dictionary containing edge data.
//         attr : keyword arguments, optional
//             Edge data (or labels or objects) can be assigned using
//             keyword arguments.
//
//         See Also
//         --------
//         add_edge : add a single edge
//         add_weighted_edges_from : convenient way to add weighted edges
//
//         Notes
//         -----
//         Adding the same edge twice has no effect but any edge data
//         will be updated when each duplicate edge is added.
//
//         Edge attributes specified in an ebunch take precedence over
//         attributes specified via keyword arguments.
//
//         When adding edges from an iterator over the graph you are changing,
//         a `RuntimeError` can be raised with message:
//         `RuntimeError: dictionary changed size during iteration`. This
//         happens when the graph's underlying dictionary is modified during
//         iteration. To avoid this error, evaluate the iterator into a separate
//         object, e.g. by using `list(iterator_of_edges)`, and pass this
//         object to `G.add_edges_from`.
//
//         Examples
//         --------
//         >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc
//         >>> G.add_edges_from([(0, 1), (1, 2)])  # using a list of edge tuples
//         >>> e = zip(range(0, 3), range(1, 4))
//         >>> G.add_edges_from(e)  # Add the path graph 0-1-2-3
//
//         Associate data to edges
//
//         >>> G.add_edges_from([(1, 2), (2, 3)], weight=3)
//         >>> G.add_edges_from([(3, 4), (1, 4)], label="WN2898")
//
//         Evaluate an iterator over a graph if using it to modify the same graph
//
//         >>> G = nx.Graph([(1, 2), (2, 3), (3, 4)])
//         >>> # Grow graph by one new node, adding edges to all existing nodes.
//         >>> # wrong way - will raise RuntimeError
//         >>> # G.add_edges_from(((5, n) for n in G.nodes))
//         >>> # correct way - note that there will be no self-edge for node 5
//         >>> G.add_edges_from(list((5, n) for n in G.nodes))
//         """
//         for e in ebunch_to_add:
//             ne = len(e)
//             if ne == 3:
//                 u, v, dd = e
//             elif ne == 2:
//                 u, v = e
//                 dd = {}  # doesn't need edge_attr_dict_factory
//             else:
//                 raise NetworkXError(f"Edge tuple {e} must be a 2-tuple or 3-tuple.")
//             if u not in self._node:
//                 if u is None:
//                     raise ValueError("None cannot be a node")
//                 self._adj[u] = self.adjlist_inner_dict_factory()
//                 self._node[u] = self.node_attr_dict_factory()
//             if v not in self._node:
//                 if v is None:
//                     raise ValueError("None cannot be a node")
//                 self._adj[v] = self.adjlist_inner_dict_factory()
//                 self._node[v] = self.node_attr_dict_factory()
//             datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
//             datadict.update(attr)
//             datadict.update(dd)
//             self._adj[u][v] = datadict
//             self._adj[v][u] = datadict

// def _dijkstra_multisource(
//     G, sources, weight, pred=None, paths=None, cutoff=None, target=None
// ):
//     """Uses Dijkstra's algorithm to find shortest weighted paths
//
//     Parameters
//     ----------
//     G : NetworkX graph
//
//     sources : non-empty iterable of nodes
//         Starting nodes for paths. If this is just an iterable containing
//         a single node, then all paths computed by this function will
//         start from that node. If there are two or more nodes in this
//         iterable, the computed paths may begin from any one of the start
//         nodes.
//
//     weight: function
//         Function with (u, v, data) input that returns that edge's weight
//         or None to indicate a hidden edge
//
//     pred: dict of lists, optional(default=None)
//         dict to store a list of predecessors keyed by that node
//         If None, predecessors are not stored.
//
//     paths: dict, optional (default=None)
//         dict to store the path list from source to each node, keyed by node.
//         If None, paths are not stored.
//
//     target : node label, optional
//         Ending node for path. Search is halted when target is found.
//
//     cutoff : integer or float, optional
//         Length (sum of edge weights) at which the search is stopped.
//         If cutoff is provided, only return paths with summed weight <= cutoff.
//
//     Returns
//     -------
//     distance : dictionary
//         A mapping from node to shortest distance to that node from one
//         of the source nodes.
//
//     Raises
//     ------
//     NodeNotFound
//         If any of `sources` is not in `G`.
//
//     Notes
//     -----
//     The optional predecessor and path dictionaries can be accessed by
//     the caller through the original pred and paths objects passed
//     as arguments. No need to explicitly return pred or paths.
//
//     """
//     G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)
//
//     push = heappush
//     pop = heappop
//     dist = {}  # dictionary of final distances
//     seen = {}
//     # fringe is heapq with 3-tuples (distance,c,node)
//     # use the count c to avoid comparing nodes (may not be able to)
//     c = count()
//     fringe = []
//     for source in sources:
//         seen[source] = 0
//         push(fringe, (0, next(c), source))
//     while fringe:
//         (d, _, v) = pop(fringe)
//         if v in dist:
//             continue  # already searched this node.
//         dist[v] = d
//         if v == target:
//             break
//         for u, e in G_succ[v].items():
//             cost = weight(v, u, e)
//             if cost is None:
//                 continue
//             vu_dist = dist[v] + cost
//             if cutoff is not None:
//                 if vu_dist > cutoff:
//                     continue
//             if u in dist:
//                 u_dist = dist[u]
//                 if vu_dist < u_dist:
//                     raise ValueError("Contradictory paths found:", "negative weights?")
//                 elif pred is not None and vu_dist == u_dist:
//                     pred[u].append(v)
//             elif u not in seen or vu_dist < seen[u]:
//                 seen[u] = vu_dist
//                 push(fringe, (vu_dist, next(c), u))
//                 if paths is not None:
//                     paths[u] = paths[v] + [u]
//                 if pred is not None:
//                     pred[u] = [v]
//             elif vu_dist == seen[u]:
//                 if pred is not None:
//                     pred[u].append(v)
//
//     # The optional predecessor and path dictionaries can be accessed
//     # by the caller via the pred and paths objects passed as arguments.
//     return dist

use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap, HashSet};

// implement Ord and PartialOrd for DistCountNode
// NOTE: compare based on the distance first, then the count, and finally the node.
// reverse the comparison to make it a min heap.
// The count is used to pop FIFO when the distance is the same.

#[derive(Debug)]
struct DistCountNode(f64, u32, u32);
impl Ord for DistCountNode {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap()
            .reverse()
            .then_with(|| self.1.cmp(&other.1).reverse())
            .then_with(|| self.2.cmp(&other.2).reverse())
    }
}
impl PartialOrd for DistCountNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Eq for DistCountNode {}
impl PartialEq for DistCountNode {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1 && self.2 == other.2
    }
}

// Undirected graph data structure:
// Graph has node that stores node numbers, and adj that stores the adjacency list of the graph.
// The adjacency list is a hashmap that stores the weight of the edge between two nodes.
pub struct Graph {
    pub node: HashSet<u32>,
    pub adj: HashMap<u32, HashMap<u32, f64>>,
}

// Implementation of the Graph data structure.
// The Graph data structure has the following methods:
// 1. add_weighted_edges_from (only allow 3-tuple and not 2-tuple)
// 2. dijkstra_multisource

impl Graph {
    pub fn new() -> Graph {
        Graph {
            node: HashSet::new(),
            adj: HashMap::new(),
        }
    }

    pub fn add_weighted_edges_from(&mut self, ebunch_to_add: Vec<(u32, u32, f64)>) {
        for e in ebunch_to_add {
            let (u, v, dd) = e;
            if !self.node.contains(&u) {
                self.adj.insert(u, HashMap::new());
                self.node.insert(u);
            }
            if !self.node.contains(&v) {
                self.adj.insert(v, HashMap::new());
                self.node.insert(v);
            }
            self.adj.get_mut(&u).unwrap().insert(v, dd);
            self.adj.get_mut(&v).unwrap().insert(u, dd);
        }
    }

    pub fn dijkstra_singlesource(&self, source: u32, cutoff: Option<f64>) -> HashMap<u32, f64> {
        let mut dist = HashMap::new();
        let mut seen = HashMap::<u32, f64>::new();

        // fringe is heapq with 3-tuples (distance,c,node)
        // use the count c to avoid comparing nodes (may not be able to)
        let mut fringe = BinaryHeap::<DistCountNode>::new();
        let mut c: u32 = 0;

        seen.insert(source, 0.0);
        fringe.push(DistCountNode(0.0, c, source));
        c += 1;

        while !fringe.is_empty() {
            let dist_count_node = fringe.pop().unwrap();
            let (d, v) = (dist_count_node.0, dist_count_node.2);
            if dist.contains_key(&v) {
                continue;
            }
            dist.insert(v, d);

            for (u, e) in self.adj.get(&v).unwrap().iter() {
                let cost = *e;
                let vu_dist = dist.get(&v).unwrap() + cost;
                if let Some(cutoff) = cutoff {
                    if vu_dist > cutoff {
                        continue;
                    }
                }

                if dist.contains_key(u) {
                    let u_dist = *dist.get(u).unwrap();
                    if vu_dist < u_dist {
                        panic!("Contradictory paths found: negative weights?");
                    }
                } else if !seen.contains_key(u) || vu_dist < *seen.get(u).unwrap() {
                    seen.insert(*u, vu_dist);
                    fringe.push(DistCountNode(vu_dist, c, *u));
                    c += 1;
                }
            }
        }

        // remove source - source distance (0.0). maybe improve performance?
        // dist.remove(&source);

        // for consistency with networkx, add source - source distance (0.0)
        dist.insert(source, 0.0);
        dist
    }

    pub fn all_pairs_dijkstra_path_length(
        &self,
        cutoff: Option<f64>,
    ) -> HashMap<u32, HashMap<u32, f64>> {
        // sequential version

        // let mut dist = HashMap::new();
        // for u in self.node.iter() {
        //     dist.insert(*u, self.dijkstra_singlesource(*u, cutoff));
        // }

        // parallel version
        self.node
            .par_iter()
            .map(|u| (*u, self.dijkstra_singlesource(*u, cutoff)))
            .collect()
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[pyfunction]
fn all_pairs_dijkstra_path_length(edges: Vec<(u32, u32, f64)>, cutoff: Option<f64>) -> Py<PyAny> {
    let mut graph = Graph::new();
    graph.add_weighted_edges_from(edges);

    let dist = graph.all_pairs_dijkstra_path_length(cutoff);
    Python::with_gil(|py| dist.to_object(py))
}

/// A Python module implemented in Rust.
#[pymodule]
fn rust_graph(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(all_pairs_dijkstra_path_length, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BinaryHeap;

    use crate::DistCountNode;

    #[test]
    fn test_minheap() {
        let mut heap = BinaryHeap::new();
        heap.push(DistCountNode(1.0, 1, 1));
        heap.push(DistCountNode(5.0, 2, 3));
        heap.push(DistCountNode(3.0, 4, 5));
        assert_eq!(heap.pop(), Some(DistCountNode(1.0, 1, 1)));
        heap.push(DistCountNode(0.0, 4, 5));
        assert_eq!(heap.pop(), Some(DistCountNode(0.0, 4, 5)));
    }
}
