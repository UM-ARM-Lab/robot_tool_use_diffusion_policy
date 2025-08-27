import h5py
import argparse

def print_h5_tree(file_path, max_items=10, max_depth=None):
    """
    Print the hierarchical structure of an H5 file
    
    Args:
        file_path: Path to the H5 file
        max_items: Maximum number of items to show per group (default: 10)
        max_depth: Maximum depth to traverse (default: None for unlimited)
    """
    
    def print_item(name, obj, level=0, current_depth=0):
        """Recursively print items in the H5 file"""
        if max_depth is not None and current_depth > max_depth:
            return
            
        indent = "  " * level
        
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name}/ (Group)")
            
            # Get all items in the group
            items = list(obj.items())
            # Show all items
            for i, (key, item) in enumerate(items):
                print_item(key, item, level + 1, current_depth + 1)
            
            # Show count if there are more items
            # if len(items) > max_items:
            #     print(f"{indent}  ... and {len(items) - max_items} more items")
                
        elif isinstance(obj, h5py.Dataset):
            shape_str = str(obj.shape)
            dtype_str = str(obj.dtype)
            
            # Get some sample data info
            size_mb = obj.size * obj.dtype.itemsize / (1024 * 1024)

            print(f"{indent}{name} (Dataset): shape={shape_str} ({dtype_str})")
            
            # Special handling for episode_ends - print the actual values
            if name == "episode_ends":
                try:
                    values = obj[:]
                    print(f"{indent}  Values: {values}")
                except Exception as e:
                    print(f"{indent}  Error reading values: {e}")

            # Show attributes if any
            if obj.attrs:
                for attr_name, attr_value in obj.attrs.items():
                    print(f"{indent}  @{attr_name}: {attr_value}")
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"H5 File: {file_path}")
            print("=" * 50)
            
            # Print root level attributes
            if f.attrs:
                print("Root Attributes:")
                for attr_name, attr_value in f.attrs.items():
                    print(f"  @{attr_name}: {attr_value}")
                print()
            
            # Print the tree structure
            for key, item in f.items():
                print_item(key, item)
                
    except Exception as e:
        print(f"Error reading H5 file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Print H5 file tree structure")
    parser.add_argument("file", help="Path to H5 file")
    parser.add_argument("--max-items", type=int, default=10, 
                       help="Maximum items to show per group (default: 10)")
    parser.add_argument("--max-depth", type=int, default=None,
                       help="Maximum depth to traverse (default: unlimited)")
    
    args = parser.parse_args()
    print_h5_tree(args.file, args.max_items, args.max_depth)

if __name__ == "__main__":
    main()