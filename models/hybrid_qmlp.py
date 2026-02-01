import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
import numpy as np
import colorsys
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random

class SphereColorDataset(Dataset):
    """
    Generates a synthetic dataset of points on a sphere with corresponding RGB colors.
    """

    def __init__(self, N):
        self.N = N
        self.points = self._fibonacci_sphere(N)
        self.colors = self._compute_colors(self.points)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        pt = torch.from_numpy(self.points[idx]).float()
        col = torch.from_numpy(self.colors[idx]).float()
        return pt, col

    @staticmethod
    def _fibonacci_sphere(N):
        pts = np.zeros((N, 3), dtype=np.float32)
        offset = 2.0 / N
        increment = np.pi * (3.0 - np.sqrt(5))
        for i in range(N):
            y = ((i * offset) - 1) + (offset / 2)
            r = np.sqrt(1 - y * y)
            phi = (i * increment) % (2 * np.pi)
            x = np.cos(phi) * r
            z = np.sin(phi) * r
            pts[i] = (x, y, z)
        return pts

    @staticmethod
    def _compute_colors(pts):
        N = pts.shape[0]
        cols = np.zeros((N, 3), dtype=np.float32)
        for i, (x, y, z) in enumerate(pts):
            theta = np.arctan2(y, x) % (2 * np.pi)
            hue = theta / (2 * np.pi)
            sat = (z + 1) / 2
            val = 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
            cols[i] = (r, g, b)
        return cols


class QLayer(tq.QuantumModule):
    """
    A quantum layer (ansatz) with trainable single-qubit rotations
    and fixed CNOT entanglement.
    """

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        self.rzs = nn.ModuleList([tq.RZ(has_params=True, trainable=True) for _ in range(n_wires)])
        self.rys = nn.ModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        # Operators for functional mode (stateless)
        self.rz_func = tq.RZ(has_params=False)
        self.ry_func = tq.RY(has_params=False)

    def forward(self, qdev: tq.QuantumDevice, params=None):
        if params is None:
            # Standard forward pass using internal parameters
            for i in range(self.n_wires):
                self.rzs[i](qdev, wires=i)
                self.rys[i](qdev, wires=i)
        else:
            # Functional forward pass using provided parameters
            current_pos = 0
            for i in range(self.n_wires):
                # RZ
                self.rz_func(qdev, wires=i, params=params[:, current_pos])
                current_pos += 1
                # RY
                self.ry_func(qdev, wires=i, params=params[:, current_pos])
                current_pos += 1
        
        for i in range(self.n_wires - 1):
            qdev.cnot(wires=[i, i + 1])
        if self.n_wires > 1:
            qdev.cnot(wires=[self.n_wires - 1, 0])

class BatchedQLayer(tq.QuantumModule):
    """
    A batched quantum layer (ansatz) with trainable single-qubit rotations
    and fixed CNOT entanglement.
    """
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires
        # Use operators without internal parameters for functional application
        self.rz = tq.RZ(has_params=False)
        self.ry = tq.RY(has_params=False)


    def forward(self, qdev: tq.QuantumDevice, params: torch.Tensor):
        # params has shape [bsz, 2 * n_wires]
        current_pos = 0
        for i in range(self.n_wires):
            # RZ
            self.rz(qdev, wires=i, params=params[:, current_pos])
            current_pos += 1
            # RY
            self.ry(qdev, wires=i, params=params[:, current_pos])
            current_pos += 1

        for i in range(self.n_wires - 1):
            qdev.cnot(wires=[i, i + 1])
        if self.n_wires > 1:
            qdev.cnot(wires=[self.n_wires - 1, 0])

        qdev.reset_op_history()


class SphericalEncodingLayer(tq.QuantumModule):
    """
    A quantum encoding layer that maps 3D Cartesian coordinates (x,y,z) on a sphere
    to spherical angles (theta, phi) and uses them to set the state of the qubits.
    """

    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        # assert x.abs().max() < 1.
        theta = torch.acos(x[:, 2]).unsqueeze(-1)
        phi = torch.atan2(x[:, 1], x[:, 0]).unsqueeze(-1)
        phi = torch.fmod(phi + 2 * np.pi, 2 * np.pi)
        for i in range(self.n_wires):
            qdev.ry(wires=i, params=theta)
            qdev.rz(wires=i, params=phi)

class AngleEncodingLayer(tq.QuantumModule):
    def __init__(self, n_wires: int):
        super().__init__()
        self.n_wires = n_wires

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor):
        assert x.abs().max() <= 1.
        assert x.shape[1] == self.n_wires
        for i in range(self.n_wires):
            qdev.ry(wires=i, params=x[:, i].unsqueeze(-1))


class HybridQuantumMLP(nn.Module):
    def __init__(self, quantum_params: dict, mlp_params: dict, no_hyper: bool):
        super().__init__()
        self.quantum_params = quantum_params
        self.mlp_params = mlp_params
        self.no_hyper = no_hyper
        
        if quantum_params["perform_quantum"]:
            self.quantum_net = BatchedQuantumMLP(
                **{
                    "n_qubits": quantum_params["n_qubits"],
                    "n_layers": quantum_params["n_layers"],
                    "out_dim": 1,
                    "use_classical_post_processing": False,
                    "encoder": quantum_params["encoder"]
                }
            )
            self.quantum_net_params_no = self._get_quantum_param_no()
        else:
            self.quantum_net_params_no = 0
            self.quantum_net = lambda x, y: x
        
        if mlp_params["perform_mlp"]:
            self.layer_sizes = [self.quantum_params.get("n_qubits", 3)] + mlp_params["mlp_hidden_layer_sizes"] + [mlp_params["out_dim"]]
            self.mlp = lambda _input, _params: self._calculate_target_mlp_output(_input, _params, mlp_params["angle_encoding"])
            self.mlp_params_no = self._get_mlp_param_no()
        else:
            self.layer_sizes = []
            self.mlp_params_no = 0
            self.mlp = lambda x, y: x
        
        if self.no_hyper:
            self.params = nn.Parameter(torch.randn(self.mlp_params_no + self.quantum_net_params_no, requires_grad=True))
    
    def forward(self, x, batched_weights):
        if self.no_hyper:
            batched_weights = self.params.repeat(x.shape[0], 1)
        out1 = self.quantum_net(x, batched_weights[:, :self.quantum_net_params_no])
        out2 = self.mlp(out1, batched_weights[:, self.quantum_net_params_no:])
        return out2
    
    def _get_quantum_param_no(self):
        return QuantumMLP(
            n_qubits=self.quantum_params["n_qubits"],
            n_layers=self.quantum_params["n_layers"],
            out_dim=1,
            use_classical_post_processing=False,
        ).get_total_params()
    
    def _get_mlp_param_no(self):
        total_params = 0
        current_input_dim = self.layer_sizes[0]

        for i, next_layer_dim in enumerate(self.layer_sizes[1:]):
            num_weights = current_input_dim * next_layer_dim
            num_biases = next_layer_dim
            total_params += num_weights + num_biases
            current_input_dim = next_layer_dim
        return total_params
        
    
    def _calculate_target_mlp_output(self, x, weights, angle_encoding=False):
        ''' 
        Input:
            x (torch.Tensor): input to target network
            w (torch.Tensor): weights from hypernetwork
            layer_sizes (list): list of sequential layers in nn, e.g. [3, 16, 16, 1] means input 3, two hidden layers with size 16 and output with size 1
        '''
        current_input_dim = self.layer_sizes[0]
        current_param_idx = 0
        if angle_encoding:
            theta = torch.acos(x[:, 2]).unsqueeze(-1)
            phi = torch.atan2(x[:, 1], x[:, 0]).unsqueeze(-1)
            phi = torch.fmod(phi + 2 * np.pi, 2 * np.pi)
            x = torch.cat([
                torch.ones_like(theta),
                theta,
                phi
            ], dim=-1)
        current_output = x.unsqueeze(1) # bmm needs (batch, 1, features)

        for i, next_layer_dim in enumerate(self.layer_sizes[1:]):
            num_weights = current_input_dim * next_layer_dim
            num_biases = next_layer_dim

            # Weights
            layer_w_flat = weights[:, current_param_idx : current_param_idx + num_weights]
            layer_w = layer_w_flat.reshape(-1, current_input_dim, next_layer_dim)
            current_param_idx += num_weights

            # Biases
            layer_b = weights[:, current_param_idx : current_param_idx + num_biases]
            current_param_idx += num_biases

            layer_output = torch.bmm(current_output, layer_w) # (batch_size, 1, next_layer_dim)
            layer_output = layer_output + layer_b.unsqueeze(1)

            if i == (len(self.layer_sizes[1:]) - 1): # check for last layer
                current_output = F.sigmoid(layer_output) # sigmoid for last layer
            else:
                current_output = F.leaky_relu(layer_output) # ReLu for other layers

            current_input_dim = next_layer_dim

        # get rid of additional dim from bmm
        return current_output.squeeze(1).reshape(-1, self.layer_sizes[-1])

class QuantumMLP(nn.Module):
    """
    The complete Hybrid Quantum-Classical MLP model.
    This architecture was found to be the most performant in our experiments.
    """

    def __init__(self, n_qubits=3, n_layers=4, out_dim=3, encoder="spherical", use_classical_post_processing=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.out_dim = out_dim
        self.use_classical_post_processing = use_classical_post_processing
        self.encoder_type = encoder

        assert encoder in ["spherical", "angle", "none"]
        if encoder == "spherical":
            self.encoder = SphericalEncodingLayer(self.n_qubits)
        elif encoder == "angle":
            self.encoder = AngleEncodingLayer(self.n_qubits)

        self.layers = nn.ModuleList(
            [QLayer(self.n_qubits) for _ in range(self.n_layers)]
        )

        self.measure = tq.MeasureAll(tq.PauliZ)

        if self.use_classical_post_processing:
            self.classical_post = nn.Linear(n_qubits, self.out_dim)  # Simple linear readout

    def forward(self, x, weights=None):
        bsz = x.size(0)
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        qdev.reset_op_history()

        if weights is None:
            # Standard forward pass using internal parameters
            for layer in self.layers:
                layer(qdev)
            
            exp_vals = self.measure(qdev)

            if self.use_classical_post_processing:
                rgb = self.classical_post(exp_vals)
                rgb = torch.sigmoid(rgb)
            else:
                rgb = torch.sigmoid(exp_vals)
        else:
            # Functional forward pass with provided weights
            current_pos = 0
            for layer in self.layers:
                num_layer_params = 2 * layer.n_wires
                layer_weights = weights[:, current_pos : current_pos + num_layer_params]
                layer(qdev, params=layer_weights)
                current_pos += num_layer_params
            
            exp_vals = self.measure(qdev)

            if self.use_classical_post_processing:
                # Batched classical post-processing
                num_classical_w = self.classical_post.weight.numel()
                num_classical_b = self.classical_post.bias.numel()

                classical_w = weights[:, current_pos: current_pos + num_classical_w].view(bsz, self.classical_post.out_features, self.classical_post.in_features)
                current_pos += num_classical_w
                
                classical_b = weights[:, current_pos: current_pos + num_classical_b]
                current_pos += num_classical_b

                # einsum for batched matrix-vector multiplication
                rgb = torch.einsum('b...d,bd->b...', classical_w, exp_vals) + classical_b
                rgb = torch.sigmoid(rgb)
            else:
                rgb = torch.sigmoid(exp_vals)
        return rgb

    def get_total_params(self):
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def load_weights(self, weights: torch.Tensor):
        """
        Loads weights from a flat tensor into the model's parameters.

        Args:
            weights (torch.Tensor): A 1D tensor containing all weights for the model.
        """
        if weights.numel() != self.get_total_params():
            raise ValueError(f"Incorrect number of weights. Expected {self.get_total_params()}, got {weights.numel()}")

        current_pos = 0

        for param in self.parameters():
            if param.requires_grad:
                num_elements = param.numel()
                weight_slice = weights[current_pos: current_pos + num_elements]
                with torch.no_grad():
                    param.copy_(weight_slice.view(param.shape))
                current_pos += num_elements


class BatchedQuantumMLP(QuantumMLP):
    """
    A batched version of the QuantumMLP.
    """

    def __init__(self, n_qubits=3, n_layers=4, out_dim=3, use_classical_post_processing=True, encoder="spherical"):
        super().__init__(n_qubits, n_layers, out_dim, encoder, use_classical_post_processing)
        self.layers = nn.ModuleList(
            [BatchedQLayer(self.n_qubits) for _ in range(self.n_layers)]
        )
        if self.encoder_type == "none":
            del self.encoder
            self.manual_ry_gates = nn.ModuleList([tq.RY(has_params=False) for _ in range(n_qubits)])

    def forward(self, x, batched_weights=None):
        bsz = x.size(0)
        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device)
        if self.encoder_type != "none":
            self.encoder(qdev, x)
        else:
            # Use x (shared_features) directly as quantum gate parameters
            # Assuming x has shape [batch_size, 3] and we want to use it as rotation angles
            for i in range(min(self.n_qubits, x.shape[1])):
                self.manual_ry_gates[i](qdev, wires=i, params=x[:, i])
    
        qdev.reset_op_history()  # Reset history but keep the quantum state
    
        if batched_weights is not None:
            # Batched inference mode
            current_pos = 0
            for layer in self.layers:
                # Extract weights for this layer for the whole batch
                num_layer_params = 2 * layer.n_wires
                layer_params = batched_weights[:, current_pos : current_pos + num_layer_params]
                current_pos += num_layer_params
                layer(qdev, params=layer_params)
            
            exp_vals = self.measure(qdev)

            if self.use_classical_post_processing:
                # Batched classical post-processing
                num_classical_w = self.classical_post.weight.numel()
                num_classical_b = self.classical_post.bias.numel()

                classical_w = batched_weights[:, current_pos: current_pos + num_classical_w].view(bsz, self.classical_post.out_features, self.classical_post.in_features)
                current_pos += num_classical_w
                
                classical_b = batched_weights[:, current_pos: current_pos + num_classical_b]
                current_pos += num_classical_b

                # einsum for batched matrix-vector multiplication
                rgb = torch.einsum('b...d,bd->b...', classical_w, exp_vals) + classical_b
                rgb = torch.sigmoid(rgb)
            else:
                rgb = torch.sigmoid(exp_vals)
        else:
            # Standard training/inference mode
            for layer in self.layers:
                layer(qdev)
            exp_vals = self.measure(qdev)
            if self.use_classical_post_processing:
                rgb = self.classical_post(exp_vals)
                rgb = torch.sigmoid(rgb)
            else:
                rgb = torch.sigmoid(exp_vals)
        return rgb



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def train_model(model, dataloader, optimizer, device, epochs):
    model.to(device)
    model.train()
    for epoch in range(1, epochs + 1):
        loop = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", leave=True)
        for pts, colors in loop:
            pts, colors = pts.to(device), colors.to(device)
            optimizer.zero_grad()
            preds = model(pts)
            loss = F.mse_loss(preds, colors)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())


if __name__ == "__main__":
    N_POINTS = 100_000
    BATCH_SIZE = 1024
    EPOCHS = 50
    LEARNING_RATE = 1e-3
    SEED = 42

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hybrid_model = QuantumMLP(
        n_qubits=3,
        n_layers=4,
        use_classical_post_processing=True
    )
    hybrid_model.to(device)

    total_params = hybrid_model.get_total_params()
    print(f"Instantiated Hybrid QMLP with {total_params} parameters.")

    print("\n--- Hypernetwork Demonstration ---")
    hypernet_generated_weights = torch.randn(total_params, device=device)
    print(f"Generated a flat weight tensor of size: {hypernet_generated_weights.shape}")

    hybrid_model.load_weights(hypernet_generated_weights)
    print("Successfully loaded weights from the flat tensor into the model.")

    first_param_after_load = next(hybrid_model.parameters())
    print(f"First parameter value after loading: \n{first_param_after_load.data.view(-1)[:5]}...")

    print("\n--- Standard Training Demonstration ---")

    hybrid_model_for_training = QuantumMLP(
        n_qubits=3,
        n_layers=4,
        use_classical_post_processing=True
    )

    dataset = SphereColorDataset(N=N_POINTS)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(hybrid_model_for_training.parameters(), lr=LEARNING_RATE)

    train_model(hybrid_model_for_training, dataloader, optimizer, device, EPOCHS)

    print("\nTraining finished.")
