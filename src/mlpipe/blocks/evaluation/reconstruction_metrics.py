"""Evaluation metrics for reconstruction tasks.

This module provides comprehensive evaluation metrics for autoencoder and
generative models, focusing on reconstruction quality assessment.

Metrics include:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Signal-to-Noise Ratio (SNR)
- Structural Similarity Index (SSIM) for image data
- Per-sample error analysis
"""

from pathlib import Path
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim

    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

from mlpipe.core.interfaces import Evaluator
from mlpipe.core.registry import register


@register("eval.reconstruction")
class ReconstructionEvaluator(Evaluator):
    """Comprehensive reconstruction quality evaluator.

    Designed for autoencoders, variational autoencoders, and other
    generative models that reconstruct input data.

    Features:
    - Multiple reconstruction metrics (MSE, MAE, RMSE, SNR)
    - Per-sample error analysis
    - Optional visualization generation
    - Support for various data types (tabular, image)

    Example usage:
        evaluator = ReconstructionEvaluator()
        evaluator.build({
            'metrics': ['mse', 'mae', 'rmse', 'snr'],
            'per_sample': True,
            'plot_reconstruction': True
        })
        results = evaluator.evaluate(original_data, reconstructed_data)
    """

    def __init__(self):
        super().__init__()
        self.config = {}

    def build(self, config: Dict[str, Any]) -> None:
        """Configure the reconstruction evaluator."""
        default_config = {
            "metrics": [
                "mse",
                "mae",
                "rmse",
                "snr",
            ],  # Available: 'mse', 'mae', 'rmse', 'snr', 'ssim'
            "per_sample": True,  # Compute per-sample errors
            "plot_reconstruction": False,  # Generate reconstruction plots
            "save_outputs": False,  # Save reconstructed samples
            "output_dir": "reconstruction_outputs",
            "plot_samples": 5,  # Number of samples to plot
            "verbose": True,
            "epsilon": 1e-8,  # Small value to avoid division by zero
        }

        self.config = {**default_config, **config}

        # Validate metrics
        available_metrics = ["mse", "mae", "rmse", "snr", "ssim"]
        invalid_metrics = [m for m in self.config.get["metrics"] if m not in available_metrics]
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}. Available: {available_metrics}")

        if "ssim" in self.config.get["metrics"] and not SSIM_AVAILABLE:
            print(
                "⚠️  Warning: SSIM metric requires scikit-image. Install with: pip install scikit-image"
            )
            self.config.get["metrics"] = [m for m in self.config.get["metrics"] if m != "ssim"]

        if self.config.get("verbose", True):
            print("🔍 Reconstruction Evaluator Configuration:")
            print(f"   Metrics: {self.config.get('metrics', ['mse', 'mae', 'rmse'])}")
            print(f"   Per-sample analysis: {self.config.get('per_sample', True)}")
            print(f"   Generate plots: {self.config.get('plot_reconstruction', True)}")

    def _compute_mse(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Compute Mean Squared Error."""
        error = (original - reconstructed) ** 2
        if self.config.get("per_sample", True):
            return np.mean(
                error, axis=tuple(range(1, error.ndim))
            )  # Mean over all dims except batch
        else:
            return np.mean(error)

    def _compute_mae(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Compute Mean Absolute Error."""
        error = np.abs(original - reconstructed)
        if self.config.get("per_sample", True):
            return np.mean(error, axis=tuple(range(1, error.ndim)))
        else:
            return np.mean(error)

    def _compute_rmse(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Compute Root Mean Squared Error."""
        mse = self._compute_mse(original, reconstructed)
        return np.sqrt(mse)

    def _compute_snr(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Compute Signal-to-Noise Ratio in dB."""
        signal_power = (
            np.mean(original**2, axis=tuple(range(1, original.ndim)))
            if self.config.get("per_sample", True)
            else np.mean(original**2)
        )
        noise_power = (
            np.mean((original - reconstructed) ** 2, axis=tuple(range(1, original.ndim)))
            if self.config.get("per_sample", True)
            else np.mean((original - reconstructed) ** 2)
        )

        # Avoid division by zero
        noise_power = np.maximum(noise_power, self.config.get["epsilon"])

        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear + self.config.get["epsilon"])

        return snr_db

    def _compute_ssim(
        self, original: np.ndarray, reconstructed: np.ndarray
    ) -> Union[float, np.ndarray]:
        """Compute Structural Similarity Index (for image data)."""
        if not SSIM_AVAILABLE:
            raise ValueError("SSIM requires scikit-image. Install with: pip install scikit-image")

        # SSIM is typically used for 2D images
        if original.ndim < 2:
            # For 1D data, reshape to 2D
            side_length = int(np.sqrt(original.shape[-1]))
            if side_length**2 == original.shape[-1]:
                original_2d = original.reshape(-1, side_length, side_length)
                reconstructed_2d = reconstructed.reshape(-1, side_length, side_length)
            else:
                print("⚠️  Warning: Cannot compute SSIM for 1D data that's not a perfect square")
                return np.nan
        else:
            original_2d = original
            reconstructed_2d = reconstructed

        if self.config.get("per_sample", True):
            ssim_values = []
            for i in range(original_2d.shape[0]):
                # Ensure data is in proper range for SSIM
                orig_img = original_2d[i]
                recon_img = reconstructed_2d[i]

                # Normalize to [0, 1] if needed
                if orig_img.max() > 1.0 or orig_img.min() < 0.0:
                    orig_img = (orig_img - orig_img.min()) / (
                        orig_img.max() - orig_img.min() + self.config.get["epsilon"]
                    )
                    recon_img = (recon_img - recon_img.min()) / (
                        recon_img.max() - recon_img.min() + self.config.get["epsilon"]
                    )

                try:
                    ssim_val = ssim(orig_img, recon_img, data_range=1.0)
                    ssim_values.append(ssim_val)
                except Exception as e:
                    print(f"⚠️  Warning: Could not compute SSIM for sample {i}: {e}")
                    ssim_values.append(np.nan)

            return np.array(ssim_values)
        else:
            # Global SSIM (average over all samples)
            try:
                # Flatten spatial dimensions for global computation
                orig_flat = original_2d.reshape(-1, *original_2d.shape[-2:])
                recon_flat = reconstructed_2d.reshape(-1, *reconstructed_2d.shape[-2:])

                # Compute average SSIM
                ssim_values = []
                for i in range(min(100, orig_flat.shape[0])):  # Limit to 100 samples for efficiency
                    orig_img = orig_flat[i]
                    recon_img = recon_flat[i]

                    if orig_img.max() > 1.0 or orig_img.min() < 0.0:
                        orig_img = (orig_img - orig_img.min()) / (
                            orig_img.max() - orig_img.min() + self.config.get["epsilon"]
                        )
                        recon_img = (recon_img - recon_img.min()) / (
                            recon_img.max() - recon_img.min() + self.config.get["epsilon"]
                        )

                    ssim_val = ssim(orig_img, recon_img, data_range=1.0)
                    ssim_values.append(ssim_val)

                return np.mean(ssim_values)
            except Exception as e:
                print(f"⚠️  Warning: Could not compute SSIM: {e}")
                return np.nan

    def _plot_reconstructions(
        self, original: np.ndarray, reconstructed: np.ndarray, output_dir: Path
    ) -> None:
        """Generate reconstruction comparison plots."""
        output_dir.mkdir(parents=True, exist_ok=True)

        n_samples = min(self.config.get["plot_samples"], original.shape[0])

        if original.ndim == 2:  # Tabular data or flattened images
            # For tabular data, plot feature comparisons
            fig, axes = plt.subplots(n_samples, 1, figsize=(12, 2 * n_samples))
            if n_samples == 1:
                axes = [axes]

            for i in range(n_samples):
                axes[i].plot(original[i], label="Original", alpha=0.7, linewidth=2)
                axes[i].plot(reconstructed[i], label="Reconstructed", alpha=0.7, linewidth=2)
                axes[i].set_title(f"Sample {i+1} - Reconstruction Comparison")
                axes[i].set_xlabel("Feature Index")
                axes[i].set_ylabel("Value")
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

        elif original.ndim == 3:  # Image data (samples, height, width)
            # Assume grayscale images
            fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
            if n_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(n_samples):
                # Original
                axes[i, 0].imshow(original[i], cmap="gray")
                axes[i, 0].set_title(f"Original {i+1}")
                axes[i, 0].axis("off")

                # Reconstructed
                axes[i, 1].imshow(reconstructed[i], cmap="gray")
                axes[i, 1].set_title(f"Reconstructed {i+1}")
                axes[i, 1].axis("off")

                # Difference
                diff = np.abs(original[i] - reconstructed[i])
                axes[i, 2].imshow(diff, cmap="hot")
                axes[i, 2].set_title(f"Absolute Difference {i+1}")
                axes[i, 2].axis("off")

        elif original.ndim == 4:  # Color images (samples, height, width, channels)
            fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
            if n_samples == 1:
                axes = axes.reshape(1, -1)

            for i in range(n_samples):
                # Original
                axes[i, 0].imshow(original[i])
                axes[i, 0].set_title(f"Original {i+1}")
                axes[i, 0].axis("off")

                # Reconstructed
                axes[i, 1].imshow(np.clip(reconstructed[i], 0, 1))
                axes[i, 1].set_title(f"Reconstructed {i+1}")
                axes[i, 1].axis("off")

                # Difference
                diff = np.mean(np.abs(original[i] - reconstructed[i]), axis=2)
                axes[i, 2].imshow(diff, cmap="hot")
                axes[i, 2].set_title(f"Mean Absolute Difference {i+1}")
                axes[i, 2].axis("off")

        plt.tight_layout()
        plot_path = output_dir / "reconstruction_comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()

        if self.config.get["verbose"]:
            print(f"📊 Reconstruction plots saved to: {plot_path}")

    def _save_samples(
        self, original: np.ndarray, reconstructed: np.ndarray, output_dir: Path
    ) -> None:
        """Save original and reconstructed samples."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as numpy arrays
        np.save(output_dir / "original_samples.npy", original[: self.config.get["plot_samples"]])
        np.save(
            output_dir / "reconstructed_samples.npy", reconstructed[: self.config.get["plot_samples"]]
        )

        if self.config.get["verbose"]:
            print(f"💾 Samples saved to: {output_dir}")

    def evaluate(self, original: np.ndarray, reconstructed: np.ndarray, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Evaluate reconstruction quality.

        Args:
            original: Original input data
            reconstructed: Reconstructed output data  
            config: Additional configuration parameters (optional)

        Returns:
            Dictionary containing computed metrics
        """
        # Update config if provided
        if config:
            self.config.update(config)
            
        # Convert to numpy arrays if needed
        if hasattr(original, 'values'):
            original = original.values
        if hasattr(reconstructed, 'values'):
            reconstructed = reconstructed.values
            
        # Ensure numpy arrays
        original = np.array(original)
        reconstructed = np.array(reconstructed)
            
        if original.shape != reconstructed.shape:
            raise ValueError(
                f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
            )

        if self.config.get("verbose", True):
            print("🔍 Evaluating reconstruction quality:")
            print(f"   Data shape: {original.shape}")
            print(f"   Computing metrics: {self.config.get('metrics', ['mse', 'mae', 'rmse'])}")

        results = {}

        # Compute requested metrics
        metrics = self.config.get("metrics", ["mse", "mae", "rmse"])
        for metric in metrics:
            if metric == "mse":
                results["mse"] = self._compute_mse(original, reconstructed)
            elif metric == "mae":
                results["mae"] = self._compute_mae(original, reconstructed)
            elif metric == "rmse":
                results["rmse"] = self._compute_rmse(original, reconstructed)
            elif metric == "snr":
                results["snr"] = self._compute_snr(original, reconstructed)
            elif metric == "ssim":
                results["ssim"] = self._compute_ssim(original, reconstructed)

        # Compute summary statistics if per_sample is True
        if self.config.get("per_sample", False):
            summary_results = {}
            for metric, values in results.items():
                if isinstance(values, np.ndarray) and values.size > 1:
                    summary_results[f"{metric}_mean"] = np.mean(values)
                    summary_results[f"{metric}_std"] = np.std(values)
                    summary_results[f"{metric}_min"] = np.min(values)
                    summary_results[f"{metric}_max"] = np.max(values)
                    summary_results[f"{metric}_median"] = np.median(values)
                else:
                    summary_results[metric] = values

            results.update(summary_results)

        # Generate visualizations if requested
        if self.config.get("plot_reconstruction", False):
            output_dir = Path(self.config.get("output_dir", "."))
            self._plot_reconstructions(original, reconstructed, output_dir)

        # Save samples if requested
        if self.config.get("save_outputs", False):
            output_dir = Path(self.config.get("output_dir", "."))
            self._save_samples(original, reconstructed, output_dir)

        if self.config.get("verbose", True):
            print("✅ Reconstruction evaluation complete:")
            for metric, value in results.items():
                if isinstance(value, np.ndarray) and value.size > 1:
                    print(f"   {metric}: {np.mean(value):.6f} ± {np.std(value):.6f}")
                elif isinstance(value, (int, float)):
                    print(f"   {metric}: {value:.6f}")

        return results
