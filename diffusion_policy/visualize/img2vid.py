import yaml
import logging
from pathlib import Path
from typing import Dict
import cv2
import argparse

class ImageToVideoConverter:
    """
    Utility for converting image sequences to videos.
    Handles datasets with structured image folders and creates aligned videos.
    """
    
    def __init__(self, config_path: str = "configs/infrastructure.yaml"):
        """Initialize the converter with configuration."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {config_path} not found. Using default settings.")
            return {"video_settings": {"fps": 24, "codec": "mp4v"}}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the converter."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger('ImageToVideoConverter')
    
    def find_continuous_sequence_length(self, imgs_path: Path) -> int:
        """Find the shortest continuous sequence length across all image folders."""
        if not imgs_path.exists():
            self.logger.error(f"Images path {imgs_path} does not exist")
            return 0
        
        # Find all subdirectories (depth, edges, rgb, etc.)
        subdirs = [d for d in imgs_path.iterdir() if d.is_dir()]
        if not subdirs:
            self.logger.warning(f"No subdirectories found in {imgs_path}")
            return 0
        
        min_continuous_length = float('inf')
        
        for subdir in subdirs:
            # Find all numbered files in this subdirectory
            numbered_files = []
            for file_path in subdir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    # Extract number from filename (format like "depth_0000.png")
                    stem = file_path.stem
                    if '_' in stem:
                        try:
                            number = int(stem.split('_')[-1])
                            numbered_files.append(number)
                        except ValueError:
                            continue
            
            if not numbered_files:
                self.logger.warning(f"No numbered image files found in {subdir}")
                min_continuous_length = 0
                break
            
            # Find continuous sequence starting from 0
            numbered_files.sort()
            continuous_length = 0
            
            if numbered_files[0] == 0:
                for i, num in enumerate(numbered_files):
                    if num == i:
                        continuous_length = i + 1
                    else:
                        break
            
            min_continuous_length = min(min_continuous_length, continuous_length)
            self.logger.info(f"Folder {subdir.name}: continuous sequence 0-{continuous_length-1}")
        
        if min_continuous_length == float('inf'):
            min_continuous_length = 0
        
        self.logger.info(f"Shortest continuous sequence: 0-{min_continuous_length-1}")
        return min_continuous_length
    
    def create_video_from_images(self, image_dir: Path, output_path: Path, 
                                sequence_length: int, fps: int = 24) -> bool:
        """Create video from numbered image sequence."""
        if sequence_length <= 0:
            self.logger.warning(f"No valid sequence length for {image_dir}")
            return False
        
        # Find the image format and naming pattern
        sample_files = list(image_dir.glob("*_0000.*"))
        if not sample_files:
            self.logger.warning(f"No files matching pattern *_0000.* in {image_dir}")
            return False
        
        sample_file = sample_files[0]
        prefix = sample_file.stem.rsplit('_', 1)[0]
        extension = sample_file.suffix
        
        # Load first image to get dimensions
        try:
            first_image_path = image_dir / f"{prefix}_0000{extension}"
            first_image = cv2.imread(str(first_image_path))
            if first_image is None:
                self.logger.error(f"Could not load first image: {first_image_path}")
                return False
            
            height, width, _ = first_image.shape
        except Exception as e:
            self.logger.error(f"Error loading first image: {e}")
            return False
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            self.logger.error(f"Could not open video writer for {output_path}")
            return False
        
        missing_images = []
        try:
            # Write frames to video
            for i in range(sequence_length):
                image_path = image_dir / f"{prefix}_{i:04d}{extension}"
                
                if not image_path.exists():
                    missing_images.append(str(image_path))
                    continue
                
                image = cv2.imread(str(image_path))
                if image is None:
                    missing_images.append(str(image_path))
                    continue
                
                # Resize if dimensions don't match
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height))
                
                video_writer.write(image)
            
            video_writer.release()
            
            if missing_images:
                self.logger.warning(f"Missing {len(missing_images)} images for {image_dir.name}: {missing_images[:5]}{'...' if len(missing_images) > 5 else ''}")
            
            self.logger.info(f"Created video: {output_path} ({sequence_length} frames, {len(missing_images)} missing)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating video {output_path}: {e}")
            video_writer.release()
            return False
    
    def process_camera_dataset(self, camera_path: Path, fps: int = 24) -> bool:
        """Process a camera dataset to create videos from image sequences."""
        if not camera_path.exists():
            self.logger.error(f"Camera path {camera_path} does not exist")
            return False
        
        imgs_path = camera_path / "imgs"
        videos_path = camera_path / "videos"
        
        if not imgs_path.exists():
            self.logger.error(f"Images path {imgs_path} does not exist")
            return False
        
        # Create videos directory if it doesn't exist
        videos_path.mkdir(exist_ok=True)
        
        # Find the shortest continuous sequence across all image folders
        sequence_length = self.find_continuous_sequence_length(imgs_path)
        
        if sequence_length <= 0:
            self.logger.error(f"No valid continuous sequence found in {imgs_path}")
            return False
        
        # Process each image subdirectory
        subdirs = [d for d in imgs_path.iterdir() if d.is_dir()]
        success_count = 0
        total_missing = 0
        
        for subdir in subdirs:
            video_name = f"{subdir.name}.mp4"
            output_path = videos_path / video_name
            
            self.logger.info(f"Creating video for {subdir.name}...")
            
            if self.create_video_from_images(subdir, output_path, sequence_length, fps):
                success_count += 1
            else:
                self.logger.error(f"Failed to create video for {subdir.name}")
        
        if success_count != len(subdirs):
            self.logger.warning(f"Not all images were rendered due to misalignment. Successfully created {success_count}/{len(subdirs)} videos.")
        
        self.logger.info(f"Successfully created {success_count}/{len(subdirs)} videos in {videos_path}")
        return success_count > 0
    
    def process_all_datasets(self, base_path: str, fps: int = 24) -> bool:
        """Process all datasets to create videos from image sequences."""
        base_path = Path(base_path).expanduser()
        
        if not base_path.exists():
            self.logger.error(f"Base path {base_path} does not exist")
            return False
        
        # Find all datasets (directories containing camera subdirectories)
        dataset_dirs = []
        for item in base_path.rglob("*/camera"):
            if item.is_dir():
                dataset_dirs.append(item.parent)
        
        if not dataset_dirs:
            self.logger.warning(f"No datasets with camera directories found in {base_path}")
            return False
        
        success_count = 0
        total_datasets = len(dataset_dirs)
        
        self.logger.info(f"Processing {total_datasets} datasets...")
        
        for dataset_dir in dataset_dirs:
            camera_dir = dataset_dir / "camera"
            self.logger.info(f"Processing dataset: {dataset_dir.name}")
            
            if self.process_camera_dataset(camera_dir, fps):
                success_count += 1
            else:
                self.logger.error(f"Failed to process dataset: {dataset_dir.name}")
        
        self.logger.info(f"Successfully processed {success_count}/{total_datasets} datasets")
        return success_count > 0
    
    def process_single_camera(self, camera_path: str, fps: int = 24) -> bool:
        """Process a single camera directory."""
        camera_path = Path(camera_path).expanduser()
        return self.process_camera_dataset(camera_path, fps)

def main():
    """Main function for image to video conversion."""
    parser = argparse.ArgumentParser(description="Image to Video Converter")
    parser.add_argument("--path", required=True,
                       help="Path to camera directory or base path containing datasets")
    parser.add_argument("--fps", type=int, default=24,
                       help="Frame rate for generated videos")
    parser.add_argument("--config", default="configs/infrastructure.yaml",
                       help="Path to infrastructure configuration file")
    parser.add_argument("--single-camera", action="store_true",
                       help="Process a single camera directory instead of searching for datasets")
    
    args = parser.parse_args()
    
    # Get FPS from config if available
    converter = ImageToVideoConverter(config_path=args.config)
    config_fps = converter.config.get("video_settings", {}).get("fps", 24)
    fps = args.fps if args.fps != 24 else config_fps
    
    if args.single_camera:
        # Process single camera directory
        success = converter.process_single_camera(args.path, fps=fps)
    else:
        # Process all datasets in base path
        success = converter.process_all_datasets(args.path, fps=fps)
    
    if success:
        converter.logger.info("Video conversion completed successfully")
    else:
        converter.logger.error("Video conversion failed")
        exit(1)

if __name__ == "__main__":
    main()
