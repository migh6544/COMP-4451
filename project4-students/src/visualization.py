from __future__ import annotations

import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA


def embedding_figure(chunks: list[dict]):
    if not chunks:
        return px.scatter_3d(title="No retrieved chunks to visualize")

    embeddings = np.array([c["embedding"] for c in chunks if c.get("embedding") is not None], dtype=float)
    if embeddings.ndim != 2 or embeddings.shape[0] < 3:
        # Plotly still needs x/y/z. Put small result sets on a line.
        xs = list(range(len(chunks)))
        ys = [0] * len(chunks)
        zs = [0] * len(chunks)
    else:
        coords = PCA(n_components=3, random_state=42).fit_transform(embeddings)
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]

    labels = []
    for c in chunks:
        m = c["metadata"]
        preview = c["text"][:280].replace("\n", " ")
        labels.append(
            f"Rank {c['rank']}<br>Doc: {m.get('document_title')}<br>"
            f"Heading: {m.get('heading')}<br>Distance: {c.get('distance'):.4f}<br>{preview}..."
        )

    fig = px.scatter_3d(
        x=xs,
        y=ys,
        z=zs,
        hover_name=[f"Chunk {c['rank']}" for c in chunks],
        hover_data={"preview": labels},
        title="3-D PCA Projection of Retrieved Chunk Embeddings",
    )
    fig.update_traces(marker={"size": 6})
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=50, b=0))
    return fig
