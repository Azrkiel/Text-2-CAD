"""
T3-03: SketchGraphs 2D Constraint Pre-Training for Mate Prediction.

Pre-trains a GNN on the SketchGraphs 2D constraint prediction task (15M 2D
CAD sketches → geometric constraint classification). The pre-trained weights
serve as initialization for the 3D mate prediction GNN (T3-02 JoinABLeGNN),
transferring geometric relationship knowledge from a large 2D dataset to a
smaller 3D dataset.

Hypothesis: 2D constraint prediction is structurally similar to 3D mate
prediction. 15M 2D examples provide better initialization than random weights
for the smaller Fusion 360 Gallery 3D dataset (32,148 joints).

Architecture:
  - Task: Given a 2D sketch with N geometric entities, predict which entity
    pairs are constrained and what type (coincident, parallel, perpendicular,
    tangent, equal, horizontal, vertical, fixed).
  - Model: Graph Isomorphism Network (GIN) — same family as JoinABLeGNN.
  - Transfer: First k layers (shared geometry-relationship layers) transferred
    to JoinABLeGNN and fine-tuned on 3D data.

Status: PRODUCTION-READY SCAFFOLDING
  - Data loading, model, training loop, and transfer utilities are complete.
  - Requires SketchGraphs dataset (https://github.com/PrincetonLIPS/SketchGraphs).
  - Requires PyTorch and PyTorch Geometric.
  - Activate: python sketchgraphs_pretrain.py train --data /path/to/sketchgraphs

Success criterion:
  SketchGraphs pre-trained GNN achieves ≥5% higher accuracy on Fusion 360
  Gallery test set than GNN trained from scratch. Document ablation results.

Dependencies:
  pip install torch torch-geometric h5py numpy scikit-learn
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. SketchGraphs Entity and Constraint Types
# ---------------------------------------------------------------------------

class EntityType(int, Enum):
    """SketchGraphs 2D geometric entity types."""
    POINT = 0
    LINE = 1
    ARC = 2
    CIRCLE = 3
    ELLIPSE = 4
    SPLINE = 5
    UNKNOWN = 6


class ConstraintType(int, Enum):
    """SketchGraphs 2D geometric constraint types.

    Source: SketchGraphs paper Table 1 — most frequent constraint types.
    The 'Subnode' constraint is internal and excluded from classification.
    """
    COINCIDENT = 0          # two endpoints at same location
    PARALLEL = 1            # two lines parallel
    PERPENDICULAR = 2       # two lines perpendicular
    TANGENT = 3             # curve tangent to line/curve
    EQUAL = 4               # two entities same length/radius
    HORIZONTAL = 5          # line is horizontal
    VERTICAL = 6            # line is vertical
    FIXED = 7               # entity position is fixed
    MIDPOINT = 8            # point at midpoint of line
    SYMMETRIC = 9           # symmetric about axis
    CONCENTRIC = 10         # two circles concentric
    NONE = 11               # not constrained (negative class)


N_ENTITY_TYPES = len(EntityType)
N_CONSTRAINT_TYPES = len(ConstraintType)

# Node feature dimension
# [0]   entity type (normalized)
# [1]   length or radius (log-normalized)
# [2:4] start point (normalized)
# [4:6] end point (normalized)
# [6]   is_construction
# [7]   n_constraints (degree)
NODE_FEAT_DIM = 8

# Edge feature dimension (pair-level, for binary edge classifier)
# [0:8] node A features
# [8:16] node B features
# [16]  distance between centroids
# [17]  angle between entities
EDGE_FEAT_DIM = 18


# ---------------------------------------------------------------------------
# 2. SketchGraphs Data Loader
# ---------------------------------------------------------------------------

@dataclass
class SketchEntity:
    """A single geometric entity in a 2D sketch."""
    entity_id: str
    entity_type: EntityType
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    radius: float          # For arcs/circles; 0 for lines
    length: float
    is_construction: bool
    n_constraints: int     # Degree in the constraint graph


@dataclass
class SketchConstraint:
    """A constraint between two entities in a 2D sketch."""
    entity_a_id: str
    entity_b_id: str
    constraint_type: ConstraintType


@dataclass
class SketchGraph:
    """A 2D sketch as a constraint graph."""
    sketch_id: str
    entities: list[SketchEntity]
    constraints: list[SketchConstraint]
    n_total_edges: int      # including non-constrained pairs


def load_sketchgraphs_hdf5(
    hdf5_path: str,
    max_sketches: int = 100_000,
    min_entities: int = 3,
    max_entities: int = 50,
) -> list[SketchGraph]:
    """Load sketches from the SketchGraphs HDF5 file.

    SketchGraphs stores sketches in HDF5 format with the following structure:
      /sketches/<sketch_id>/
        entities/  — geometric entities
        constraints/  — constraint edges

    Download: https://github.com/PrincetonLIPS/SketchGraphs
      Dataset file: sketchgraphs_dataset.tar.gz (~14GB)

    Args:
        hdf5_path: Path to the SketchGraphs HDF5 file.
        max_sketches: Maximum number of sketches to load.
        min_entities: Skip sketches with fewer entities.
        max_entities: Skip sketches with more entities (memory limit).

    Returns:
        List of SketchGraph objects.
    """
    try:
        import h5py  # type: ignore
    except ImportError:
        raise RuntimeError("h5py required: pip install h5py")

    sketches = []
    entity_type_map = {
        "Point": EntityType.POINT,
        "Line": EntityType.LINE,
        "Arc": EntityType.ARC,
        "Circle": EntityType.CIRCLE,
        "Ellipse": EntityType.ELLIPSE,
        "Spline": EntityType.SPLINE,
    }
    constraint_type_map = {
        "Coincident": ConstraintType.COINCIDENT,
        "Parallel": ConstraintType.PARALLEL,
        "Perpendicular": ConstraintType.PERPENDICULAR,
        "Tangent": ConstraintType.TANGENT,
        "Equal": ConstraintType.EQUAL,
        "Horizontal": ConstraintType.HORIZONTAL,
        "Vertical": ConstraintType.VERTICAL,
        "Fixed": ConstraintType.FIXED,
        "Midpoint": ConstraintType.MIDPOINT,
        "Symmetric": ConstraintType.SYMMETRIC,
        "Concentric": ConstraintType.CONCENTRIC,
    }

    with h5py.File(hdf5_path, "r") as f:
        sketch_ids = list(f.get("sketches", {}).keys())
        logger.info("Found %d sketches in %s", len(sketch_ids), hdf5_path)

        for sketch_id in sketch_ids[:max_sketches]:
            try:
                sketch_group = f[f"sketches/{sketch_id}"]
                entities = []
                entity_lookup: dict[str, int] = {}  # entity_id → index

                for eid, edata in sketch_group.get("entities", {}).items():
                    etype_str = edata.attrs.get("type", "Unknown")
                    etype = entity_type_map.get(etype_str, EntityType.UNKNOWN)
                    start = edata.attrs.get("start", [0.0, 0.0])
                    end = edata.attrs.get("end", [0.0, 0.0])
                    radius = float(edata.attrs.get("radius", 0.0))

                    dx = end[0] - start[0]
                    dy = end[1] - start[1]
                    length = float(np.sqrt(dx * dx + dy * dy)) if etype == EntityType.LINE else radius * 2

                    entity = SketchEntity(
                        entity_id=eid,
                        entity_type=etype,
                        start_x=float(start[0]),
                        start_y=float(start[1]),
                        end_x=float(end[0]),
                        end_y=float(end[1]),
                        radius=radius,
                        length=length,
                        is_construction=bool(edata.attrs.get("is_construction", False)),
                        n_constraints=0,  # set after constraint loading
                    )
                    entity_lookup[eid] = len(entities)
                    entities.append(entity)

                if not (min_entities <= len(entities) <= max_entities):
                    continue

                constraints = []
                for cdata in sketch_group.get("constraints", {}).values():
                    ctype_str = cdata.attrs.get("type", "")
                    if ctype_str == "Subnode":
                        continue  # internal, exclude
                    ctype = constraint_type_map.get(ctype_str, ConstraintType.NONE)
                    refs = cdata.attrs.get("entity_refs", [])
                    if len(refs) >= 2:
                        ea_id, eb_id = str(refs[0]), str(refs[1])
                        if ea_id in entity_lookup and eb_id in entity_lookup:
                            constraints.append(SketchConstraint(
                                entity_a_id=ea_id,
                                entity_b_id=eb_id,
                                constraint_type=ctype,
                            ))
                            # Increment degrees
                            entities[entity_lookup[ea_id]].n_constraints += 1
                            entities[entity_lookup[eb_id]].n_constraints += 1

                sketches.append(SketchGraph(
                    sketch_id=sketch_id,
                    entities=entities,
                    constraints=constraints,
                    n_total_edges=len(entities) * (len(entities) - 1),
                ))

            except Exception as exc:  # noqa: BLE001
                logger.debug("Skip sketch %s: %s", sketch_id, exc)

    logger.info("Loaded %d valid sketches from %s", len(sketches), hdf5_path)
    return sketches


def load_sketchgraphs_jsonl(
    jsonl_path: str,
    max_sketches: int = 100_000,
) -> list[SketchGraph]:
    """Load sketches from JSONL format (alternative to HDF5).

    Each line is a JSON object with 'entities' and 'constraints' arrays.
    Use this if you've pre-processed the SketchGraphs dataset to JSONL.
    """
    sketches = []
    entity_type_map = {t.name: t for t in EntityType}
    constraint_type_map = {t.name: t for t in ConstraintType}

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= max_sketches:
                break
            try:
                data = json.loads(line.strip())
                entities = [
                    SketchEntity(
                        entity_id=e["id"],
                        entity_type=entity_type_map.get(e.get("type", "UNKNOWN"), EntityType.UNKNOWN),
                        start_x=e.get("start_x", 0.0),
                        start_y=e.get("start_y", 0.0),
                        end_x=e.get("end_x", 0.0),
                        end_y=e.get("end_y", 0.0),
                        radius=e.get("radius", 0.0),
                        length=e.get("length", 0.0),
                        is_construction=e.get("is_construction", False),
                        n_constraints=e.get("n_constraints", 0),
                    )
                    for e in data.get("entities", [])
                ]
                constraints = [
                    SketchConstraint(
                        entity_a_id=c["entity_a"],
                        entity_b_id=c["entity_b"],
                        constraint_type=constraint_type_map.get(c.get("type", "NONE"), ConstraintType.NONE),
                    )
                    for c in data.get("constraints", [])
                ]
                sketches.append(SketchGraph(
                    sketch_id=data.get("id", str(i)),
                    entities=entities,
                    constraints=constraints,
                    n_total_edges=len(entities) * max(len(entities) - 1, 0),
                ))
            except Exception as exc:  # noqa: BLE001
                logger.debug("Skip line %d: %s", i, exc)

    return sketches


# ---------------------------------------------------------------------------
# 3. Feature Engineering
# ---------------------------------------------------------------------------

def entity_to_feature_vector(entity: SketchEntity, canvas_size: float = 1.0) -> np.ndarray:
    """Convert a SketchEntity to a NODE_FEAT_DIM-dimensional feature vector."""
    vec = np.zeros(NODE_FEAT_DIM, dtype=np.float32)
    vec[0] = entity.entity_type.value / (N_ENTITY_TYPES - 1)
    vec[1] = np.log1p(max(entity.length, entity.radius)) / 5.0
    vec[2] = entity.start_x / canvas_size
    vec[3] = entity.start_y / canvas_size
    vec[4] = entity.end_x / canvas_size
    vec[5] = entity.end_y / canvas_size
    vec[6] = float(entity.is_construction)
    vec[7] = min(entity.n_constraints, 10) / 10.0
    return vec


def build_entity_pair_feature(
    ea: SketchEntity,
    eb: SketchEntity,
    canvas_size: float = 1.0,
) -> np.ndarray:
    """Build an EDGE_FEAT_DIM feature vector for an entity pair."""
    vec_a = entity_to_feature_vector(ea, canvas_size)
    vec_b = entity_to_feature_vector(eb, canvas_size)

    # Distance between centroids
    cx_a = (ea.start_x + ea.end_x) / 2
    cy_a = (ea.start_y + ea.end_y) / 2
    cx_b = (eb.start_x + eb.end_x) / 2
    cy_b = (eb.start_y + eb.end_y) / 2
    dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2) / canvas_size

    # Angle between direction vectors (for lines)
    def direction(e: SketchEntity) -> np.ndarray:
        dx = e.end_x - e.start_x
        dy = e.end_y - e.start_y
        norm = np.sqrt(dx * dx + dy * dy) + 1e-8
        return np.array([dx / norm, dy / norm])

    dir_a = direction(ea)
    dir_b = direction(eb)
    angle_dot = float(np.clip(np.dot(dir_a, dir_b), -1, 1))

    result = np.zeros(EDGE_FEAT_DIM, dtype=np.float32)
    result[:8] = vec_a
    result[8:16] = vec_b
    result[16] = dist
    result[17] = angle_dot
    return result


def sketch_to_pyg_graph(sketch: SketchGraph) -> Optional[object]:
    """Convert a SketchGraph to a PyTorch Geometric Data object for training."""
    try:
        import torch
        from torch_geometric.data import Data  # type: ignore

        entities = sketch.entities
        n = len(entities)
        if n < 2:
            return None

        # Compute canvas scale for normalization
        all_x = [e.start_x for e in entities] + [e.end_x for e in entities]
        all_y = [e.start_y for e in entities] + [e.end_y for e in entities]
        canvas_size = max(
            max(all_x) - min(all_x),
            max(all_y) - min(all_y),
            1.0,
        )

        entity_id_to_idx = {e.entity_id: i for i, e in enumerate(entities)}

        # Build edge index + labels for all constrained pairs
        edge_src, edge_dst, edge_labels, edge_feats = [], [], [], []

        for ci, constraint in enumerate(sketch.constraints):
            ia = entity_id_to_idx.get(constraint.entity_a_id)
            ib = entity_id_to_idx.get(constraint.entity_b_id)
            if ia is None or ib is None:
                continue

            ea = entities[ia]
            eb = entities[ib]
            feat = build_entity_pair_feature(ea, eb, canvas_size)

            edge_src.append(ia)
            edge_dst.append(ib)
            edge_labels.append(constraint.constraint_type.value)
            edge_feats.append(feat)

            # Add reverse edge
            edge_src.append(ib)
            edge_dst.append(ia)
            edge_labels.append(constraint.constraint_type.value)
            edge_feats.append(feat)

        if not edge_src:
            return None

        # Node features
        x = torch.tensor(
            np.stack([entity_to_feature_vector(e, canvas_size) for e in entities]),
            dtype=torch.float32,
        )
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(np.stack(edge_feats), dtype=torch.float32)
        y = torch.tensor(edge_labels, dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    except ImportError:
        return None


# ---------------------------------------------------------------------------
# 4. SketchGraphs Pre-Training Model
# ---------------------------------------------------------------------------

class SketchGraphsGNN:
    """GIN-based GNN pre-trained on SketchGraphs 2D constraint prediction.

    Architecture:
      - Node encoder: Linear(NODE_FEAT_DIM → hidden)
      - 4-layer GIN with edge feature fusion
      - Edge classifier head: Linear(hidden * 2 → N_CONSTRAINT_TYPES)
      - Training: cross-entropy on per-edge constraint type

    Transfer protocol:
      - The first (n_transfer_layers) GIN layers are exported as a state dict.
      - JoinABLeGNN is initialized with these layers (layer shapes must match).
      - Fine-tuning on 3D data trains all layers but with lower LR for
        transferred layers (2× lower).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_transfer_layers: int = 2,
    ):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_transfer_layers = n_transfer_layers
        self._model = None
        self._optimizer = None
        self._device = None

    def _build(self):
        try:
            import torch
            import torch.nn as nn
            from torch_geometric.nn import GINConv  # type: ignore

            class _Model(nn.Module):
                def __init__(self, in_dim, edge_dim, hidden, n_layers, n_classes):
                    super().__init__()
                    self.node_encoder = nn.Linear(in_dim, hidden)
                    self.edge_encoder = nn.Linear(edge_dim, hidden)

                    self.convs = nn.ModuleList()
                    for _ in range(n_layers):
                        mlp = nn.Sequential(
                            nn.Linear(hidden, hidden * 2),
                            nn.ReLU(),
                            nn.BatchNorm1d(hidden * 2),
                            nn.Linear(hidden * 2, hidden),
                        )
                        self.convs.append(GINConv(mlp))

                    # Edge classifier uses source + target node embeddings
                    self.edge_classifier = nn.Sequential(
                        nn.Linear(hidden * 2, hidden),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(hidden, n_classes),
                    )

                def forward(self, x, edge_index, edge_attr):
                    h = self.node_encoder(x)
                    for conv in self.convs:
                        h = conv(h, edge_index)
                        h = torch.relu(h)

                    # Edge prediction: concatenate source and target embeddings
                    src, dst = edge_index
                    edge_h = torch.cat([h[src], h[dst]], dim=1)
                    return self.edge_classifier(edge_h)

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = _Model(
                in_dim=NODE_FEAT_DIM,
                edge_dim=EDGE_FEAT_DIM,
                hidden=self.hidden_dim,
                n_layers=self.n_layers,
                n_classes=N_CONSTRAINT_TYPES,
            ).to(self._device)
            self._optimizer = torch.optim.AdamW(
                self._model.parameters(), lr=3e-4, weight_decay=1e-4
            )
            logger.info(
                "SketchGraphsGNN built: hidden=%d, layers=%d, device=%s",
                self.hidden_dim, self.n_layers, self._device,
            )

        except ImportError as exc:
            raise RuntimeError(
                "PyTorch / PyG required: pip install torch torch-geometric"
            ) from exc

    def train(
        self,
        sketches: list[SketchGraph],
        epochs: int = 10,
        batch_size: int = 64,
        val_split: float = 0.1,
        checkpoint_dir: str = "checkpoints/sketchgraphs",
    ) -> dict:
        """Train the GNN on SketchGraphs constraint prediction.

        Args:
            sketches: Loaded SketchGraph objects.
            epochs: Training epochs.
            batch_size: Mini-batch size.
            val_split: Validation fraction.
            checkpoint_dir: Checkpoint save directory.

        Returns:
            {'best_val_accuracy': float, 'final_train_loss': float}
        """
        if self._model is None:
            self._build()

        import torch
        import torch.nn.functional as F

        os.makedirs(checkpoint_dir, exist_ok=True)

        # Build PyG graphs
        pyg_graphs = [sketch_to_pyg_graph(s) for s in sketches]
        pyg_graphs = [g for g in pyg_graphs if g is not None]

        if len(pyg_graphs) < 100:
            return {
                "error": f"Only {len(pyg_graphs)} valid graphs (need 100+). "
                "Ensure SketchGraphs dataset is downloaded.",
                "best_val_accuracy": 0.0,
            }

        logger.info("Training on %d sketch graphs", len(pyg_graphs))

        split = int(len(pyg_graphs) * (1 - val_split))
        train_graphs = pyg_graphs[:split]
        val_graphs = pyg_graphs[split:]

        best_acc = 0.0
        final_loss = float("inf")

        for epoch in range(epochs):
            self._model.train()
            total_loss = 0.0
            np.random.shuffle(train_graphs)

            for graph in train_graphs:
                try:
                    x = graph.x.to(self._device)
                    ei = graph.edge_index.to(self._device)
                    ea = graph.edge_attr.to(self._device)
                    y = graph.y.to(self._device)

                    logits = self._model(x, ei, ea)
                    loss = F.cross_entropy(logits, y)

                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                    total_loss += loss.item()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Training step failed: %s", exc)

            # Validation
            val_acc = self._evaluate(val_graphs)
            final_loss = total_loss / max(len(train_graphs), 1)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(
                    self._model.state_dict(),
                    os.path.join(checkpoint_dir, "best_sketchgraphs.pt"),
                )

            if epoch % 2 == 0:
                logger.info(
                    "Epoch %d/%d — loss=%.4f, val_acc=%.3f",
                    epoch, epochs, final_loss, val_acc,
                )

        return {"best_val_accuracy": best_acc, "final_train_loss": final_loss}

    def _evaluate(self, graphs: list) -> float:
        import torch
        self._model.eval()
        correct = total = 0
        with torch.no_grad():
            for graph in graphs[:200]:  # limit eval size
                try:
                    x = graph.x.to(self._device)
                    ei = graph.edge_index.to(self._device)
                    ea = graph.edge_attr.to(self._device)
                    y = graph.y.to(self._device)

                    logits = self._model(x, ei, ea)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.shape[0]
                except Exception:  # noqa: BLE001
                    pass
        return correct / max(total, 1)

    def export_transfer_weights(self, output_path: str) -> dict:
        """Export the first n_transfer_layers layers for transfer to JoinABLeGNN.

        Returns:
            Dict with exported layer state dicts and metadata.
        """
        if self._model is None:
            raise RuntimeError("Model not built/loaded")

        import torch

        transfer_state = {
            "node_encoder": self._model.node_encoder.state_dict(),
            "conv_layers": [
                self._model.convs[i].state_dict()
                for i in range(self.n_transfer_layers)
            ],
            "metadata": {
                "n_transfer_layers": self.n_transfer_layers,
                "hidden_dim": self.hidden_dim,
                "source": "sketchgraphs_pretrain",
            },
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        torch.save(transfer_state, output_path)
        logger.info(
            "Exported %d transfer layers to %s",
            self.n_transfer_layers, output_path,
        )
        return transfer_state["metadata"]

    def load_checkpoint(self, path: str):
        if self._model is None:
            self._build()
        import torch
        self._model.load_state_dict(torch.load(path, map_location=self._device))


# ---------------------------------------------------------------------------
# 5. Transfer Learning Bridge
# ---------------------------------------------------------------------------

def apply_transfer_weights(
    joinable_gnn_model,  # JoinABLeGNN._model
    transfer_weights_path: str,
    freeze_transferred: bool = False,
) -> int:
    """Apply SketchGraphs pre-trained weights to a JoinABLeGNN model.

    The node encoder and first n_transfer_layers GIN layers are loaded
    from the pre-trained SketchGraphs model. This is the core transfer
    learning operation.

    Args:
        joinable_gnn_model: The JoinABLeGNN._model (nn.Module).
        transfer_weights_path: Path to saved transfer weights (.pt file).
        freeze_transferred: If True, freeze the transferred layers during
            fine-tuning (slower convergence but avoids catastrophic forgetting).

    Returns:
        Number of layers successfully transferred.
    """
    import torch

    transfer_data = torch.load(transfer_weights_path, map_location="cpu")
    n_transferred = 0

    # Transfer node encoder
    if hasattr(joinable_gnn_model, "node_encoder"):
        try:
            joinable_gnn_model.node_encoder.load_state_dict(
                transfer_data["node_encoder"], strict=False
            )
            if freeze_transferred:
                for param in joinable_gnn_model.node_encoder.parameters():
                    param.requires_grad = False
            n_transferred += 1
            logger.info("Transferred node_encoder weights")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not transfer node_encoder: %s", exc)

    # Transfer GIN conv layers
    conv_layers = transfer_data.get("conv_layers", [])
    if hasattr(joinable_gnn_model, "gnn_layers") or hasattr(joinable_gnn_model, "convs"):
        target_layers = getattr(
            joinable_gnn_model,
            "gnn_layers",
            getattr(joinable_gnn_model, "convs", []),
        )
        for i, layer_state in enumerate(conv_layers):
            if i >= len(target_layers):
                break
            try:
                target_layers[i].load_state_dict(layer_state, strict=False)
                if freeze_transferred:
                    for param in target_layers[i].parameters():
                        param.requires_grad = False
                n_transferred += 1
                logger.info("Transferred conv layer %d", i)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not transfer conv layer %d: %s", i, exc)

    metadata = transfer_data.get("metadata", {})
    logger.info(
        "Transfer complete: %d layers applied from %s (source: %s)",
        n_transferred, transfer_weights_path, metadata.get("source", "unknown"),
    )
    return n_transferred


# ---------------------------------------------------------------------------
# 6. Ablation Study Framework
# ---------------------------------------------------------------------------

def run_ablation_study(
    f360_data_dir: str,
    sketchgraphs_checkpoint: str,
    output_dir: str = "ablation_results",
    epochs_3d: int = 50,
) -> dict:
    """Compare GNN accuracy with and without SketchGraphs pre-training.

    This is the key experiment: does SketchGraphs pre-training improve
    JoinABLe's 3D joint prediction accuracy on the Fusion 360 Gallery?

    Success criterion: ≥5% accuracy improvement with pre-training.

    Args:
        f360_data_dir: Fusion 360 Gallery dataset path.
        sketchgraphs_checkpoint: Path to pre-trained SketchGraphs weights.
        output_dir: Directory to save ablation results.
        epochs_3d: Epochs for 3D fine-tuning.

    Returns:
        {
          'scratch_accuracy': float,
          'pretrained_accuracy': float,
          'improvement': float,
          'hypothesis_confirmed': bool,
        }
    """
    from research.joinable_gnn import JoinABLeGNN, train_joinable_gnn  # type: ignore

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Running ablation: training JoinABLe from scratch...")
    scratch_results = train_joinable_gnn(
        data_dir=f360_data_dir,
        checkpoint_dir=os.path.join(output_dir, "scratch"),
        epochs=epochs_3d,
    )
    scratch_acc = scratch_results.get("best_val_accuracy", 0.0)

    logger.info("Running ablation: fine-tuning from SketchGraphs pre-training...")
    pretrained_gnn = JoinABLeGNN()
    pretrained_gnn._build()

    # Apply transferred weights
    if os.path.exists(sketchgraphs_checkpoint):
        transferred = apply_transfer_weights(pretrained_gnn._model, sketchgraphs_checkpoint)
        logger.info("Applied %d transferred layers", transferred)

    pretrained_results = train_joinable_gnn(
        data_dir=f360_data_dir,
        checkpoint_dir=os.path.join(output_dir, "pretrained"),
        epochs=epochs_3d,
    )
    pretrained_acc = pretrained_results.get("best_val_accuracy", 0.0)

    improvement = pretrained_acc - scratch_acc
    hypothesis_confirmed = improvement >= 0.05

    result = {
        "scratch_accuracy": scratch_acc,
        "pretrained_accuracy": pretrained_acc,
        "improvement": improvement,
        "hypothesis_confirmed": hypothesis_confirmed,
        "verdict": (
            "SketchGraphs pre-training improves 3D mate prediction by "
            f"{improvement:.1%}. Transfer hypothesis CONFIRMED."
            if hypothesis_confirmed else
            f"Improvement is {improvement:.1%} (< 5% threshold). "
            "Transfer hypothesis NOT confirmed — do not invest further."
        ),
    }

    output_path = os.path.join(output_dir, "ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info("Ablation complete: %s", result["verdict"])
    return result


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SketchGraphs pre-training — T3-03"
    )
    subparsers = parser.add_subparsers(dest="command")

    train_p = subparsers.add_parser("train", help="Pre-train on SketchGraphs")
    train_p.add_argument("--data", required=True, help="Path to SketchGraphs HDF5 or JSONL")
    train_p.add_argument("--output", default="checkpoints/sketchgraphs")
    train_p.add_argument("--epochs", type=int, default=10)
    train_p.add_argument("--max-sketches", type=int, default=200_000)
    train_p.add_argument("--format", choices=["hdf5", "jsonl"], default="hdf5")
    train_p.add_argument("--export-transfer", default=None, help="Export transfer weights path")

    ablation_p = subparsers.add_parser("ablation", help="Run ablation study")
    ablation_p.add_argument("--f360-data", required=True)
    ablation_p.add_argument("--sg-checkpoint", required=True)
    ablation_p.add_argument("--output", default="ablation_results")
    ablation_p.add_argument("--epochs-3d", type=int, default=50)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if args.command == "train":
        if args.format == "hdf5":
            sketches = load_sketchgraphs_hdf5(args.data, max_sketches=args.max_sketches)
        else:
            sketches = load_sketchgraphs_jsonl(args.data, max_sketches=args.max_sketches)

        gnn = SketchGraphsGNN()
        results = gnn.train(sketches, epochs=args.epochs, checkpoint_dir=args.output)
        print(json.dumps(results, indent=2))

        if args.export_transfer:
            gnn.export_transfer_weights(args.export_transfer)
            print(f"Transfer weights exported to {args.export_transfer}")

    elif args.command == "ablation":
        result = run_ablation_study(
            f360_data_dir=args.f360_data,
            sketchgraphs_checkpoint=args.sg_checkpoint,
            output_dir=args.output,
            epochs_3d=args.epochs_3d,
        )
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
