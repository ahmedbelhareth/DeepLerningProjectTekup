"""
Model Optimization - Export ONNX et optimisation du serving

Ce module fournit:
- Export au format ONNX
- Quantification des modèles
- Benchmark de performance
- Optimisation pour l'inférence
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime


class ModelOptimizer:
    """
    Optimiseur de modèles pour le déploiement.

    Fonctionnalités:
    - Export ONNX
    - Quantification (dynamique, statique)
    - Pruning
    - Benchmark de performance
    """

    def __init__(
        self,
        output_dir: str = "optimized_models"
    ):
        """
        Initialise l'optimiseur.

        Args:
            output_dir: Répertoire pour les modèles optimisés
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_onnx(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32),
        output_path: Optional[str] = None,
        opset_version: int = 13,
        dynamic_axes: Optional[Dict] = None
    ) -> str:
        """
        Exporte le modèle au format ONNX.

        Args:
            model: Modèle PyTorch
            input_shape: Forme de l'entrée
            output_path: Chemin de sortie
            opset_version: Version de l'opset ONNX
            dynamic_axes: Axes dynamiques pour les dimensions variables

        Returns:
            Chemin du fichier ONNX
        """
        if output_path is None:
            output_path = self.output_dir / f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.onnx"
        else:
            output_path = Path(output_path)

        model.eval()
        device = next(model.parameters()).device

        # Créer une entrée exemple
        dummy_input = torch.randn(*input_shape, device=device)

        # Axes dynamiques par défaut (batch size variable)
        if dynamic_axes is None:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }

        # Export ONNX
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes
        )

        # Vérifier le modèle exporté
        try:
            import onnx
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print(f"ONNX model exported and verified: {output_path}")
        except ImportError:
            print(f"ONNX model exported: {output_path} (onnx package not available for verification)")

        return str(output_path)

    def quantize_dynamic(
        self,
        model: nn.Module,
        output_path: Optional[str] = None
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Applique la quantification dynamique au modèle.

        Args:
            model: Modèle PyTorch
            output_path: Chemin de sortie

        Returns:
            Tuple (modèle quantifié, rapport)
        """
        model.eval()
        model_cpu = model.cpu()

        # Quantification dynamique
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )

        # Calculer la réduction de taille
        original_size = self._get_model_size(model_cpu)
        quantized_size = self._get_model_size(quantized_model)

        report = {
            "original_size_mb": original_size,
            "quantized_size_mb": quantized_size,
            "size_reduction": f"{(1 - quantized_size/original_size)*100:.1f}%",
            "quantization_type": "dynamic",
            "dtype": "int8"
        }

        if output_path:
            torch.save(quantized_model.state_dict(), output_path)
            report["saved_path"] = output_path

        return quantized_model, report

    def _get_model_size(self, model: nn.Module) -> float:
        """Calcule la taille du modèle en MB."""
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / (1024 * 1024)
        return size_mb

    def benchmark_inference(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32),
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Benchmark les performances d'inférence.

        Args:
            model: Modèle à benchmarker
            input_shape: Forme de l'entrée
            num_iterations: Nombre d'itérations
            warmup_iterations: Itérations de warmup

        Returns:
            Rapport de benchmark
        """
        model.eval()
        device = next(model.parameters()).device

        dummy_input = torch.randn(*input_shape, device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup_iterations):
                _ = model(dummy_input)

        # Synchroniser si CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.perf_counter()
                _ = model(dummy_input)

                if device.type == 'cuda':
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

        latencies = np.array(latencies)

        return {
            "device": str(device),
            "input_shape": input_shape,
            "num_iterations": num_iterations,
            "mean_latency_ms": float(np.mean(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "min_latency_ms": float(np.min(latencies)),
            "max_latency_ms": float(np.max(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p90_latency_ms": float(np.percentile(latencies, 90)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "throughput_fps": float(1000 / np.mean(latencies) * input_shape[0])
        }

    def benchmark_batch_sizes(
        self,
        model: nn.Module,
        batch_sizes: List[int] = [1, 2, 4, 8, 16, 32],
        num_iterations: int = 50
    ) -> Dict[str, Any]:
        """
        Benchmark différentes tailles de batch.

        Args:
            model: Modèle à benchmarker
            batch_sizes: Tailles de batch à tester
            num_iterations: Nombre d'itérations

        Returns:
            Rapport de benchmark par batch size
        """
        results = {}

        for batch_size in batch_sizes:
            try:
                input_shape = (batch_size, 3, 32, 32)
                benchmark = self.benchmark_inference(
                    model,
                    input_shape=input_shape,
                    num_iterations=num_iterations
                )
                results[batch_size] = benchmark
            except RuntimeError as e:
                # Probablement OOM
                results[batch_size] = {"error": str(e)}
                break

        # Trouver le batch size optimal
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            optimal_batch = max(
                valid_results.keys(),
                key=lambda k: valid_results[k]["throughput_fps"]
            )
        else:
            optimal_batch = 1

        return {
            "batch_benchmarks": results,
            "optimal_batch_size": optimal_batch,
            "optimal_throughput_fps": valid_results.get(optimal_batch, {}).get("throughput_fps", 0)
        }

    def profile_memory(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32)
    ) -> Dict[str, Any]:
        """
        Profile l'utilisation mémoire du modèle.

        Args:
            model: Modèle à profiler
            input_shape: Forme de l'entrée

        Returns:
            Rapport de mémoire
        """
        model.eval()
        device = next(model.parameters()).device

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

            initial_memory = torch.cuda.memory_allocated()

            dummy_input = torch.randn(*input_shape, device=device)

            with torch.no_grad():
                output = model(dummy_input)

            peak_memory = torch.cuda.max_memory_allocated()
            final_memory = torch.cuda.memory_allocated()

            return {
                "device": str(device),
                "initial_memory_mb": initial_memory / (1024 * 1024),
                "peak_memory_mb": peak_memory / (1024 * 1024),
                "final_memory_mb": final_memory / (1024 * 1024),
                "memory_increase_mb": (peak_memory - initial_memory) / (1024 * 1024)
            }
        else:
            return {
                "device": str(device),
                "model_size_mb": self._get_model_size(model),
                "note": "Detailed memory profiling only available on CUDA"
            }

    def optimize_for_inference(
        self,
        model: nn.Module,
        use_script: bool = True,
        use_trace: bool = False,
        input_shape: Tuple[int, ...] = (1, 3, 32, 32)
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Optimise le modèle pour l'inférence.

        Args:
            model: Modèle à optimiser
            use_script: Utiliser TorchScript via script
            use_trace: Utiliser TorchScript via trace
            input_shape: Forme de l'entrée pour le tracing

        Returns:
            Tuple (modèle optimisé, rapport)
        """
        model.eval()
        device = next(model.parameters()).device

        # Benchmark avant optimisation
        before_benchmark = self.benchmark_inference(model, input_shape)

        optimized_model = model

        if use_trace:
            dummy_input = torch.randn(*input_shape, device=device)
            optimized_model = torch.jit.trace(model, dummy_input)
            optimized_model = torch.jit.optimize_for_inference(optimized_model)
        elif use_script:
            try:
                optimized_model = torch.jit.script(model)
                optimized_model = torch.jit.optimize_for_inference(optimized_model)
            except Exception as e:
                print(f"TorchScript optimization failed: {e}")
                # Fallback: just freeze the model
                optimized_model = torch.jit.freeze(torch.jit.script(model.eval()))

        # Benchmark après optimisation
        after_benchmark = self.benchmark_inference(optimized_model, input_shape)

        speedup = before_benchmark["mean_latency_ms"] / after_benchmark["mean_latency_ms"]

        return optimized_model, {
            "optimization_method": "trace" if use_trace else "script",
            "before_latency_ms": before_benchmark["mean_latency_ms"],
            "after_latency_ms": after_benchmark["mean_latency_ms"],
            "speedup": f"{speedup:.2f}x",
            "before_throughput": before_benchmark["throughput_fps"],
            "after_throughput": after_benchmark["throughput_fps"]
        }

    def generate_optimization_report(
        self,
        model: nn.Module,
        model_name: str = "model"
    ) -> str:
        """Génère un rapport d'optimisation complet."""
        report = f"""# Model Optimization Report: {model_name}
Generated: {datetime.now().isoformat()}

## Model Information
- **Parameters:** {sum(p.numel() for p in model.parameters()):,}
- **Size:** {self._get_model_size(model):.2f} MB

## Inference Benchmark
"""
        benchmark = self.benchmark_inference(model)
        report += f"""
| Metric | Value |
|--------|-------|
| Mean Latency | {benchmark['mean_latency_ms']:.2f} ms |
| P50 Latency | {benchmark['p50_latency_ms']:.2f} ms |
| P99 Latency | {benchmark['p99_latency_ms']:.2f} ms |
| Throughput | {benchmark['throughput_fps']:.1f} FPS |

## Batch Size Analysis
"""
        batch_results = self.benchmark_batch_sizes(model)
        report += f"\n**Optimal Batch Size:** {batch_results['optimal_batch_size']}\n"
        report += f"**Max Throughput:** {batch_results['optimal_throughput_fps']:.1f} FPS\n\n"

        # Quantification
        report += "## Quantization Analysis\n"
        _, quant_report = self.quantize_dynamic(model.cpu())
        model.to(next(model.parameters()).device)

        report += f"""
| Metric | Value |
|--------|-------|
| Original Size | {quant_report['original_size_mb']:.2f} MB |
| Quantized Size | {quant_report['quantized_size_mb']:.2f} MB |
| Size Reduction | {quant_report['size_reduction']} |

## Recommendations
"""
        if benchmark['mean_latency_ms'] > 50:
            report += "- Consider using GPU acceleration\n"
        if batch_results['optimal_batch_size'] > 1:
            report += f"- Use batch size of {batch_results['optimal_batch_size']} for optimal throughput\n"
        report += "- Export to ONNX for production deployment\n"
        report += "- Consider dynamic quantization for CPU deployment\n"

        return report
