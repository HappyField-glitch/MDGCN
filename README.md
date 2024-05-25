# MDGCN: Multimodal Dual-graph Collaborative Network Serial Attentive Aggregation Mechanism for Micro-video Multi-label Classification

MDGCN integrates a dual-graph structure with a serial attentive aggregation mechanism to enhance performance in multi-label classification tasks for micro-videos. This approach captures complex inter-modal and intra-modal interactions effectively.

### Training and Testing Instructions

Follow these steps to configure and run the training or testing processes:

1. **Configure the Parameters**:
   - Navigate to the configuration file at `/libs/MMDLNetV0/mmdlnetv0_base.yaml`.
   - Update the parameters in the configuration file, including the file paths and operational modes:
     - `use_resume`: Set to `true` if you want to resume training from a checkpoint.
     - `test_only`: Set to `true` for testing mode or `false` for training mode.

2. **Run the Program**:
   - After setting up your configuration, execute `main.py` to start the training or testing process:
     ```bash
     python main.py
     ```

Ensure you have the required environment setup and dependencies installed before running the program. Adjust the paths and other parameters in the YAML configuration file according to your specific setup and requirements.
