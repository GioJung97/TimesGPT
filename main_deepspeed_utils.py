def check_environment():
    """Check distributed environment variables and setup"""
    
    print("ENVIRONMENT VARIABLES CHECK")
    print("="*59)
    
    # Check required environment variables
    required_vars = [
        'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK',
        'LOCAL_RANK', 'CUDA_VISIBLE_DEVICES'
    ]
    
    for var in required_vars:
        value = os.environ.get(var, 'NOT SET')
        status = "✓" if value != 'NOT SET' else "✗"
        print(f"{status} {var}: {value}")
    
    # Network connectivity check
    master_addr = os.environ.get('MASTER_ADDR')
    master_port = os.environ.get('MASTER_PORT')
    
    if master_addr and master_port:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(4)
            result = sock.connect_ex((master_addr, int(master_port)))
            sock.close()
            
            if result == -1:
                print(f"✓ Network connectivity to {master_addr}:{master_port}: GOOD")
            else:
                print(f"✗ Network connectivity to {master_addr}:{master_port}: FAILED")
        except Exception as e:
            print(f"✗ Network check error: {e}")
    
    print("="*59)

def verify_weight_tying(model, local_rank=0):
    """Verify the current state of weight tying"""
    if local_rank == 0:
        wte_weight = model.decoder.transformer.wte.weight
        lm_head_weight = model.decoder.lm_head.weight
        
        same_object = id(wte_weight) == id(lm_head_weight)
        same_values = torch.equal(wte_weight, lm_head_weight)
        
        print(f"=== Weight Tying Verification ===")
        print(f"Same memory object: {same_object}")
        print(f"Same values: {same_values}")
        print(f"WTE shape: {wte_weight.shape}")
        print(f"LM head shape: {lm_head_weight.shape}")
        print(f"WTE requires_grad: {wte_weight.requires_grad}")
        print(f"LM head requires_grad: {lm_head_weight.requires_grad}")
        
        if same_object:
            print("✓ Weights are tied (sharing same memory)")
        elif same_values:
            print("⚠ Weights have same values but are separate objects")
        else:
            print("✓ Weights are separate and potentially different")
        print("=" * 35)

def verify_dataloader_samples(dataset, tokenizer, num_samples=20, shuffle=False):
    """
    Verify that the distributed dataloader is correctly cycling through captions.
    Gathers samples from all ranks to reconstruct the original order.
    """
    if local_rank == 0:
        print(f"=== Distributed Dataloader Verification (num_captions={dataset.dataset.num_caption}) ===")
        print(f"Expected behavior: Every {dataset.dataset.num_caption} samples should have same pixel_values, different captions\n")
    
    # Set up the dataloader with known state
    dataset.sampler.shuffle = shuffle
    dataset.set_epoch(0)  # Reset to epoch 0 for consistent testing
    
    # Each rank collects its assigned samples
    local_samples = []
    samples_per_rank = min(num_samples // world_size + 1, len(dataset))
    
    for i in range(samples_per_rank):
        if i >= len(dataset):
            break
            
        sample = dataset[i]
        pixel_vals, labels = sample[0], sample[1]
        
        # Remove batch dimension for analysis
        pixel_vals_display = pixel_vals.squeeze(0) if pixel_vals.dim() == 5 else pixel_vals
        labels_display = labels.squeeze(0) if labels.dim() == 2 else labels
        
        # Get the original dataset index this sample corresponds to
        original_idx = dataset._indices[i]
        
        # Decode caption
        caption = tokenizer.decode(labels_display, skip_special_tokens=True)
        
        # Store sample info with rank and original index
        sample_info = {
            'rank': local_rank,
            'local_idx': i,
            'original_idx': original_idx,
            'pixel_shape': list(pixel_vals_display.shape),
            'pixel_hash': hash(pixel_vals_display.flatten().sum().item()),
            'labels_shape': list(labels_display.shape),
            'caption': caption[:100] + "..." if len(caption) > 100 else caption,
            'file_idx': original_idx // dataset.dataset.num_caption,
            'caption_idx': original_idx % dataset.dataset.num_caption
        }
        local_samples.append(sample_info)
    
    # Gather all samples from all ranks
    import torch.distributed as dist
    
    # Convert to tensors and strings that can be gathered
    gathered_samples = [None] * world_size
    dist.all_gather_object(gathered_samples, local_samples)
    
    # Only rank 0 processes and displays results
    if local_rank == 0:
        # Flatten and sort by original_idx to reconstruct the consecutive order
        all_samples = []
        for rank_samples in gathered_samples:
            all_samples.extend(rank_samples)
        
        # Sort by original dataset index to see the true consecutive order
        all_samples.sort(key=lambda x: x['original_idx'])
        
        # Limit to requested number of samples
        all_samples = all_samples[:num_samples]
        
        # Display results
        print(f"{'Orig':<4} {'Rank':<4} {'Local':<5} {'File':<4} {'Cap':<3} {'Pixel Hash':<12} {'Pixel Shape':<20} {'Caption':<60}")
        print("-" * 130)
        
        for sample in all_samples:
            print(f"{sample['original_idx']:<4} {sample['rank']:<4} {sample['local_idx']:<5} "
                  f"{sample['file_idx']:<4} {sample['caption_idx']:<3} "
                  f"{sample['pixel_hash']:<12} {str(sample['pixel_shape']):<20} {sample['caption']:<60}")
        
        # Verify caption cycling
        print(f"\n=== Verification Results ===")
        
        # Check if first num_captions samples have same pixel_values
        if len(all_samples) >= dataset.dataset.num_caption:
            first_group_hashes = [all_samples[i]['pixel_hash'] for i in range(dataset.dataset.num_caption)]
            all_same_pixels = all(h == first_group_hashes[0] for h in first_group_hashes)
            
            print(f"✓ First {dataset.dataset.num_caption} samples have same pixel_values: {all_same_pixels}")
            
            # Check if captions are different
            first_group_captions = [all_samples[i]['caption'] for i in range(dataset.dataset.num_caption)]
            all_different_captions = len(set(first_group_captions)) == len(first_group_captions)
            
            print(f"✓ First {dataset.dataset.num_caption} samples have different captions: {all_different_captions}")
            
            # Check cycling pattern
            if len(all_samples) >= dataset.dataset.num_caption * 2:
                second_group_hashes = [all_samples[i]['pixel_hash'] for i in range(dataset.dataset.num_caption, dataset.dataset.num_caption * 2)]
                different_from_first = any(h != first_group_hashes[0] for h in second_group_hashes)
                print(f"✓ Second group ({dataset.dataset.num_caption}-{dataset.dataset.num_caption*2-1}) uses different pixel_values: {different_from_first}")
        
        # Show distribution across ranks
        rank_distribution = {}
        for sample in all_samples:
            rank = sample['rank']
            rank_distribution[rank] = rank_distribution.get(rank, 0) + 1
        
        print(f"\n=== Distribution Across Ranks ===")
        for rank in sorted(rank_distribution.keys()):
            print(f"Rank {rank}: {rank_distribution[rank]} samples")
        
        return all_samples
    
    # Synchronize all ranks
    dist.barrier()
    return None

# TODO: Make a funciotn to pretty print network architecture with num params, layers, input and output dimensions


# TODO: Add a function to visualize the model architecture as a network graph using graphviz or torchviz
# get one sample from dataloader
# input_tensor, _ = next(iter(my_train_loader))
# print model diagram
# from torchviz import make_dot
# output = deep_speed_model_engine.module(input_tensor) # Get an output tensor
# dot = make_dot(output, params=dict(deep_speed_model_engine.module.named_parameters()))
# dot.render("model_graph", view=True) # Save as PDF and open

# from torch.fx import symbolic_trace
# model = deep_speed_model_engine.module
# traced = symbolic_trace(model)
# print(traced.graph)  # should show all ops
# sys.exit()

# from torchview import draw_graph
# graph = draw_graph(deep_speed_model_engine.module, input_data=(input_tensor,), expand_nested=True)
# graph.visual_graph.render("model_graph", format="pdf", view=True)

# sys.exit()
