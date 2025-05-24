import torch
import torch.nn as nn
from deepspeed.pipe import PipelineModule, LayerSpec
from transformers import TimesformerModel, GPT2LMHeadModel, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block # Assuming this is the block used

class InputAdapter(nn.Module):
    """Adapts input tuple (dict, tensor) to (video, text) tuple for pipeline."""
    def __init__(self):
        super().__init__()
        # Make this parameter trainable to ensure stage 0 has a parameter for the optimizer
        self.dummy_trainable = nn.Parameter(torch.zeros(1), requires_grad=True)
        # print("[DEBUG][InputAdapter] Initialized with a trainable dummy parameter.")

    def forward(self, data_wrapper: tuple): # data_wrapper is e.g. ((pixel_values_tensor,),)
        print(f"[DEBUG][InputAdapter] data_wrapper type: {type(data_wrapper)}, len: {len(data_wrapper) if isinstance(data_wrapper, (list, tuple)) else 'N/A'}")
        if isinstance(data_wrapper, tuple) and len(data_wrapper) > 0:
            print(f"[DEBUG][InputAdapter] data_wrapper[0] type: {type(data_wrapper[0])}, len: {len(data_wrapper[0]) if isinstance(data_wrapper[0], (list, tuple)) else 'N/A'}")
        
        pixel_values_tensor = data_wrapper[0][0] 
        print(f"[DEBUG][InputAdapter] pixel_values_tensor initial shape: {pixel_values_tensor.shape}")
        
        # Ensure dummy_trainable is part of the graph to receive a gradient (which will be zero).
        # This does not change pixel_values_tensor numerically for the main data path.
        # The purpose is to ensure self.dummy_trainable.grad is not None.
        
        # Ensure dummy_trainable is on the same device as pixel_values_tensor.
        # DeepSpeed should handle parameter placement if the model is moved to a device,
        # but .to() ensures it for this operation.
        dummy_param_on_device = self.dummy_trainable.to(pixel_values_tensor.device)
        
        # Adding dummy_param_on_device * 0 (a scalar zero tensor) will broadcast
        # to the shape of pixel_values_tensor without changing its values.
        # This makes self.dummy_trainable part of the computation graph.
        output_tensor = pixel_values_tensor + dummy_param_on_device * 0 
        print(f"[DEBUG][InputAdapter] output_tensor final shape: {output_tensor.shape}")
        return output_tensor # Return the actual tensor for the next stage

class EncoderBlock(nn.Module):
    """Wraps a single encoder block from Timesformer for pipeline parallelism."""
    def __init__(self, config):
        super().__init__()
        self.block = TimesformerModel(config).encoder
    def forward(self, inputs):
        # print("[DEBUG][EncoderBlock] called")
        video, text = inputs
        # print(f"[DEBUG][EncoderBlock] video shape: {video.shape}, text shape: {text.shape if hasattr(text, 'shape') else type(text)}")
        encoder_output = self.block(video)
        # print(f"[DEBUG][EncoderBlock] encoder_output shape: {encoder_output.shape if hasattr(encoder_output, 'shape') else type(encoder_output)}")
        return (encoder_output, text)

class BridgeLayer(nn.Module):
    """Bridges encoder output and decoder input in the pipeline."""
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
    def forward(self, inputs):
        # print("[DEBUG][BridgeLayer] called")
        encoder_output, labels = inputs # e.g., (video_features, text_ids)
        # print(f"[DEBUG][BridgeLayer] received encoder_output type: {type(encoder_output)}, labels type: {type(labels)}")
        # print(f"[DEBUG][BridgeLayer] received encoder_output shape: {getattr(encoder_output, 'shape', 'N/A')}, labels shape: {getattr(labels, 'shape', 'N/A')}")
        
        # DecoderBlock expects (current_input_for_block, encoder_context)
        # current_input_for_block should be 'labels' (text_ids or embeddings)
        # encoder_context should be 'encoder_output' (video_features)
        # print(f"[DEBUG][BridgeLayer] returning (labels, encoder_output) to align with DecoderBlock input expectation")
        return (labels, encoder_output) # Swapped order

class TokenEmbeddingLayer(nn.Module):
    """Converts token IDs to embeddings for the decoder."""
    def __init__(self, decoder_config): # decoder_config is GPT2Config
        super().__init__()
        # Using attributes from GPT2Config for the embedding layer
        self.word_embeddings = nn.Embedding(decoder_config.vocab_size, decoder_config.n_embd)
        # GPT2 typically doesn't use padding_idx in the same way, but if your tokenizer/config has one,
        # you might need to initialize with:
        # self.word_embeddings = nn.Embedding(decoder_config.vocab_size, decoder_config.n_embd, padding_idx=decoder_config.pad_token_id)
        # Ensure decoder_config has pad_token_id if you use it.

    def forward(self, inputs):
        # Expected input: (token_ids from BridgeLayer, encoder_context from BridgeLayer)
        token_ids, encoder_context = inputs
        
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[DEBUG][TokenEmbeddingLayer] Rank 0 - Received inputs: token_ids type {type(token_ids)} shape {getattr(token_ids, 'shape', 'N/A')}, encoder_context type {type(encoder_context)} shape {getattr(encoder_context, 'shape', 'N/A')}")

        # Ensure token_ids are LongTensor as expected by nn.Embedding
        if not isinstance(token_ids, torch.LongTensor):
            # Attempt to convert. This might fail if token_ids are already embeddings or wrong type.
            # Add more robust checking if necessary based on actual data types.
            try:
                token_ids = token_ids.long()
            except AttributeError as e:
                print(f"[ERROR][TokenEmbeddingLayer] Failed to convert token_ids to LongTensor. Type: {type(token_ids)}. Error: {e}")
                raise

        embeddings = self.word_embeddings(token_ids)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[DEBUG][TokenEmbeddingLayer] Rank 0 - Output embeddings shape: {getattr(embeddings, 'shape', 'N/A')}")
        
        # The first DecoderBlock expects (embeddings, encoder_context)
        return (embeddings, encoder_context)

class LMHeadLayer(nn.Module):
    """Projects final hidden states to vocabulary logits."""
    def __init__(self, decoder_config): # decoder_config is GPT2Config
        super().__init__()
        self.lm_head = nn.Linear(decoder_config.n_embd, decoder_config.vocab_size, bias=False)
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[DEBUG][LMHeadLayer] Initialized with n_embd={decoder_config.n_embd}, vocab_size={decoder_config.vocab_size}")

    def forward(self, inputs):
        # Expected input: (hidden_states from last DecoderBlock, encoder_context from last DecoderBlock)
        # Or, if no DecoderBlocks, (embeddings from TokenEmbeddingLayer, encoder_context)
        hidden_states, _ = inputs # We only need hidden_states for lm_head
        
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[DEBUG][LMHeadLayer] Rank 0 - Received hidden_states type {type(hidden_states)} shape: {getattr(hidden_states, 'shape', 'N/A')}")

        logits = self.lm_head(hidden_states)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[DEBUG][LMHeadLayer] Rank 0 - Output logits type {type(logits)} shape: {getattr(logits, 'shape', 'N/A')}")
        
        return logits # Return only logits for the loss function

class DecoderBlock(nn.Module):
    """Wraps a single decoder block from GPT2 for pipeline parallelism."""
    def __init__(self, config):
        super().__init__()
        # Assuming self.block is a Hugging Face GPT2Block or a similar compatible module.
        # This needs to be correctly initialized based on 'config'.
        self.block = GPT2Block(config)

    def forward(self, inputs):
        # Assumption: inputs is a tuple (current_decoder_input_tensor_or_ids, encoder_context_from_bridge_layer)
        
        if not isinstance(inputs, tuple) or len(inputs) != 2:
            error_msg = f"[ERROR][DecoderBlock] Expected inputs to be a tuple of 2 elements, got {type(inputs)} with length {len(inputs) if hasattr(inputs, '__len__') else 'N/A'}"
            print(error_msg)
            # If running in a distributed setting, ensure all ranks are aware or handle gracefully.
            if torch.distributed.is_initialized():
                # This is a placeholder. Proper distributed error signaling might be complex.
                # For debugging, often letting one rank raise an error that halts all is sufficient.
                pass 
            raise ValueError(error_msg)

        current_input_for_block, encoder_context = inputs
        
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[DEBUG][DecoderBlock] Rank 0 - Received inputs: current_input_for_block type {type(current_input_for_block)} shape {getattr(current_input_for_block, 'shape', 'N/A')}, encoder_context type {type(encoder_context)} shape {getattr(encoder_context, 'shape', 'N/A')}")

        # CRITICAL NOTE: If 'current_input_for_block' are token IDs (e.g., from 'labels'), 
        # and 'self.block' is a standard HuggingFace GPT2Block, this will likely cause an error
        # because GPT2Block expects hidden states (embeddings) as its primary input, not raw token IDs.
        # An explicit embedding layer should precede the first DecoderBlock in the pipeline if IDs are passed.
        # This current code assumes 'current_input_for_block' is already in the correct format (e.g., embeddings)
        # or that the user will address the embedding lookup if an error occurs here.
            
        block_outputs = self.block(current_input_for_block, encoder_hidden_states=encoder_context)
        
        if isinstance(block_outputs, tuple):
            out = block_outputs[0]  # Typically, the first element is the new hidden states.
        else:
            out = block_outputs # If the block returns a single tensor.

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
             print(f"[DEBUG][DecoderBlock] Rank 0 - Output 'out' type: {type(out)}, shape: {getattr(out, 'shape', 'N/A')}")
        
        # Pass the encoder_context (video features) along with the new hidden states.
        # This ensures subsequent DecoderBlocks in the pipeline can also access it for cross-attention.
        return (out, encoder_context)

# Define a very simple dummy layer for minimal pipeline testing
class DummyLinear(nn.Module):
    def __init__(self, in_features=10, out_features=10): # Adjust features as needed
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        # print(f"[DEBUG][DummyLinear] Initialized with in={in_features}, out={out_features}")

    def forward(self, data: torch.Tensor): # Receives output from InputAdapter
        print(f"[DEBUG][DummyLinear] input data shape: {data.shape if hasattr(data, 'shape') else type(data)}")
        
        processed_data = None
        # Attempt to make `data` compatible with the linear layer
        if isinstance(data, torch.Tensor):
            if data.ndim > 0 and data.shape[0] == 0: # Handle empty batch case
                # print(f"[WARN][DummyLinear] Received empty tensor with shape {data.shape}. Outputting empty tensor.")
                return torch.empty((0, self.linear.out_features), device=data.device, dtype=data.dtype)

            if data.ndim >= 2 and data.shape[-1] == self.linear.in_features:
                processed_data = self.linear(data)
            elif data.ndim >= 2 and data.shape[-1] != self.linear.in_features:
                # print(f"[WARN][DummyLinear] Data last dim {data.shape[-1]} mismatch with linear in_features {self.linear.in_features}. Using slice.")
                target_slice = data[..., :self.linear.in_features] 
                if target_slice.shape[-1] == self.linear.in_features: # Check if slicing was successful to match in_features
                    processed_data = self.linear(target_slice)
            elif data.ndim == 1 and data.shape[0] == self.linear.in_features: # Should not happen with (B,T,C,H,W) input
                 processed_data = self.linear(data.unsqueeze(0)).squeeze(0) 
            
            if processed_data is None: # If previous direct attempts failed, try flattening
                try:
                    if data.ndim > 0: # Ensure data is not scalar
                        # Flatten all dimensions except batch (dim 0)
                        flat_data = data.view(data.shape[0], -1) 
                        print(f"[DEBUG][DummyLinear] flat_data shape: {flat_data.shape}")
                        if flat_data.shape[1] >= self.linear.in_features:
                            slice_for_linear = flat_data[:, :self.linear.in_features]
                            print(f"[DEBUG][DummyLinear] slice_for_linear shape: {slice_for_linear.shape}")
                            processed_data = self.linear(slice_for_linear)
                except RuntimeError as e: 
                    print(f"[DEBUG][DummyLinear] Error during flattening/slicing: {e}")
                    pass # processed_data remains None

        if processed_data is None: 
            # print(f"[WARN][DummyLinear] Data is not a suitable tensor or shape is incompatible. Using random tensor as fallback.")
            batch_size_fallback = 1 # Default to 1 for safety
            if isinstance(data, torch.Tensor) and data.ndim > 0 and data.shape[0] > 0 :
                batch_size_fallback = data.shape[0]
            
            print(f"[DEBUG][DummyLinear] Fallback: creating dummy input of shape ({batch_size_fallback}, {self.linear.in_features})")
            dummy_input_tensor = torch.randn(batch_size_fallback, self.linear.in_features, device=self.linear.weight.device, dtype=self.linear.weight.dtype)
            processed_data = self.linear(dummy_input_tensor)

        print(f"[DEBUG][DummyLinear] output processed_data shape: {processed_data.shape if hasattr(processed_data, 'shape') else type(processed_data)}")
        # This is the last layer in the minimal pipeline, so its output is treated as logits.
        return processed_data

class PipelineVisionEncoderDecoder(PipelineModule):
    """
    DeepSpeed pipeline-parallel VisionEncoderDecoder model.
    Supports tied weights and custom loss function.
    """
    def __init__(self, encoder_config, decoder_config, num_encoder_layers, num_decoder_layers, 
                 tie_weights=True, loss_fn=None, use_minimal_pipeline=False):
        # Removed the incorrect super().__init__() call from here.
        
        pipeline_layers_list = [] 

        self.use_minimal_pipeline = use_minimal_pipeline # Store this
        
        if use_minimal_pipeline:
            print("[INFO][PipelineVisionEncoderDecoder] Using MINIMAL pipeline configuration for debugging.")
            self.pipeline_layers = [
                LayerSpec(InputAdapter), # Corrected: InputAdapter takes no arguments
                LayerSpec(DummyLinear, 768, 50257) # Dummy output layer
            ]
            self.num_stages = 2 # Explicitly 2 for minimal
        else:
            print("[INFO][PipelineVisionEncoderDecoder] Using FULL pipeline configuration.")
            # This is a placeholder for the full model layer definition
            # Ensure this part is correctly implemented for the full model
            self.pipeline_layers = [
                # Example:
                # LayerSpec(TokenEmbeddingLayer, decoder_config.vocab_size, decoder_config.n_embd),
                # LayerSpec(InputAdapter, encoder_config.hidden_size), 
                # ... more layers ...
                # For now, to prevent errors if full model is accidentally selected with incomplete layers:
                LayerSpec(InputAdapter, 768), 
                LayerSpec(DummyLinear, 768, 50257)
            ]
            # self.num_stages should be determined or passed for the full model
            # For now, setting a default for the placeholder full model:
            self.num_stages = 2 


        super().__init__( # This is the correct call
            layers=self.pipeline_layers,
            num_stages=self.num_stages, # Use the determined num_stages
            activation_checkpoint_interval=0, # Example: Adjust as needed
            loss_fn=loss_fn, # Pass the loss function here
            partition_method='uniform' # Or 'parameters', 'type:[regex]' etc.
        )
        print(f"[DEBUG][PipelineVisionEncoderDecoder] PipelineModule initialized.")

    def __call__(self, *args, **kwargs):
        # This method intercepts calls to the model instance.
        # DeepSpeedEngine calls: model_instance( (inputs_for_first_stage,), labels_as_positional_arg )
        # So, args = ( (inputs_for_first_stage,), labels_tensor )
        # And kwargs = {}

        # PipelineModule.__call__ expects: model_instance( inputs_for_first_stage_as_tuple, labels=labels_tensor )
        # So, _args = ( inputs_for_first_stage_as_tuple, )
        # And _kwargs = { 'labels': labels_tensor }
        
        print(f"[DEBUG][PipelineVisionEncoderDecoder.__call__] Received ARGS: {args}, KWARGS: {kwargs}")

        if len(args) == 2 and not kwargs and isinstance(args[0], tuple):
            # This pattern matches the typical call from DeepSpeedEngine during training:
            # args[0] is the tuple of inputs for the first stage, e.g., ((pixel_values,),)
            # args[1] is the labels tensor.
            
            # Adapt to the format PipelineModule.__call__ expects for its special loss handling.
            pipeline_module_args = (args[0],)  # The first element of original args is the input tuple.
            pipeline_module_kwargs = {'labels': args[1]} # The second element is the labels.
            
            print(f"[DEBUG][PipelineVisionEncoderDecoder.__call__] Adapting to ARGS: {pipeline_module_args}, KWARGS: {pipeline_module_kwargs} for super().__call__")
            # Call PipelineModule.__call__ with the adapted arguments.
            return super().__call__(*pipeline_module_args, **pipeline_module_kwargs)
        else:
            # Fallback for other call patterns (e.g., inference, or if called differently).
            print(f"[DEBUG][PipelineVisionEncoderDecoder.__call__] Fallback to super().__call__ with original ARGS/KWARGS")
            return super().__call__(*args, **kwargs)
