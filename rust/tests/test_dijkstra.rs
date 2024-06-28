use std::collections::HashMap;

use rust_graph::Graph;

#[test]
fn test_dijkstra_singlesource() {
    let mut graph = Graph::new();
    graph.add_weighted_edges_from(vec![
        (0, 1, 4.0),
        (0, 7, 8.0),
        (1, 2, 8.0),
        (1, 7, 11.0),
        (2, 3, 7.0),
        (2, 8, 2.0),
        (2, 5, 4.0),
        (3, 4, 9.0),
        (3, 5, 14.0),
        (4, 5, 10.0),
        (5, 6, 2.0),
        (6, 7, 1.0),
        (6, 8, 6.0),
        (7, 8, 7.0),
    ]);

    let source = 0;
    let result = graph.dijkstra_singlesource(source, None);
    assert_eq!(
        result,
        HashMap::<u32, f64>::from_iter(vec![
            (0, 0.0),
            (1, 4.0),
            (2, 12.0),
            (3, 19.0),
            (4, 21.0),
            (5, 11.0),
            (6, 9.0),
            (7, 8.0),
            (8, 14.0)
        ])
    );

    let source = 1;
    let result = graph.dijkstra_singlesource(source, None);
    assert_eq!(
        result,
        HashMap::<u32, f64>::from_iter(vec![
            (0, 4.0),
            (1, 0.0),
            (2, 8.0),
            (3, 15.0),
            (4, 22.0),
            (5, 12.0),
            (6, 12.0),
            (7, 11.0),
            (8, 10.0)
        ])
    );

    let source = 0;
    let result = graph.dijkstra_singlesource(source, Some(10.0));
    assert_eq!(
        result,
        HashMap::<u32, f64>::from_iter(vec![(0, 0.0), (1, 4.0), (6, 9.0), (7, 8.0),])
    );
}
