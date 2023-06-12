import plotly.graph_objects as go

# Create edges (connections between federates)
edges = [('Federate1', 'Federate2'), ('Federate1', 'Federate3'), ('Federate2', 'Federate3'),
         ('Federate2', 'Federate4'), ('Federate3', 'Federate4'), ('Federate4', 'Federate5'),
         ('Federate4', 'Federate6')]

# Create nodes
nodes = set(sum(edges, ()))

# Create a graph object
graph = go.Figure()

# Add edges
for edge in edges:
    graph.add_trace(go.Scatter(
        x=[edge[0], edge[1]],
        y=[0, 0],
        mode='lines',
        line=dict(width=1),
        hoverinfo='none'
    ))

# Add nodes
for node in nodes:
    graph.add_trace(go.Scatter(
        x=[node],
        y=[0],
        mode='markers',
        marker=dict(symbol='circle-dot', size=10, color='skyblue'),
        name=node
    ))

# Update layout
graph.update_layout(
    title='Interconnected Federates',
    showlegend=False,
    xaxis=dict(showline=False, showgrid=False, zeroline=False),
    yaxis=dict(showline=False, showgrid=False, zeroline=False, showticklabels=False),
    hovermode='closest',
)


# Show the graph
graph.write_html("thesis_fig.html")