# Troubleshooting Guide

## Common Issues

### COLMAP "No good initial image pair found"

**Cause**: Photos don't have enough feature matches. This typically happens when:
- Photos are from different objects (not the same furniture piece)
- Resolution too low (< 500px)
- Very plain/white furniture with no texture
- Images have different dimensions

**Fix**: The MVS fallback handles this automatically. To improve:
- Use higher resolution photos (720p+)
- Ensure 50%+ overlap between adjacent photos
- Avoid completely plain surfaces
- Use at least 4 photos from different angles (ideally 8+)

### "CAMERA_SINGLE_DIM_ERROR"

**Cause**: Photos have different dimensions (e.g., some portrait, some landscape).

**Fix**: The pipeline now auto-resizes all photos to 720x540 before processing. Make sure the resized images directory (`static/output/sfm/resized_images/`) has write permissions.

### Low Face Count (< 10K faces)

**Cause**: Not enough 3D points for subdivision (fewer than ~50 initial points).

**Fix**: The pipeline now supplements with dense gradient-based sampling when MVS points are insufficient. If the problem persists:
- Check `n_points` in the metadata JSON
- Ensure photos have visual texture (not plain white/black)
- Use more photos (8+ recommended)

### Low Texture Coverage (< 50%)

**Cause**: Camera projection model doesn't match the actual photo angles.

**Fix**: The texture transfer now uses per-image intrinsics and weighted blending. If coverage is still low:
- Check `texture_coverage_pct` in metadata
- More photos with overlapping views improve coverage
- Avoid extreme camera angles (> 60° from horizontal)

### Processing Time > 5 Minutes

**Cause**: COLMAP matching can be slow with many photos or very high resolution.

**Benchmark**: Expected times:
- 4 photos at 720p: ~10-15 seconds
- 8 photos at 720p: ~15-30 seconds  
- 12 photos at 720p: ~30-60 seconds

**Fix**: 
- COLMAP takes most time. Consider using `pycolmap` which runs optimized C++ code
- If pycolmap is not available, the MVS fallback is faster but less precise

### "IndexError: Out of bounds" in Texture Transfer

**Cause**: Pixel coordinates computed from projection don't fit within image dimensions.

**Fix**: This should now be fixed with per-image intrinsics and bounds checking. If it reoccurs:
- The fix ensures `0 <= px < w_i and 0 <= py < h_i` for each image's actual dimensions

### Docker Build Fails

**Cause**: Missing system dependencies for OpenCV or Python wheels.

**Fix**: The Dockerfile includes all needed system packages. If build fails:
- Ensure Docker has sufficient memory (4GB+)
- Try `docker build --no-cache -t furniture-3d-viewer .`

## Debugging

### Enable verbose logging
Add `--log-level debug` to uvicorn:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8777 --log-level debug
```

### Check processing logs
Processing output is printed to stdout with timestamps. Each stage is labeled:
- `COLMAP:` — Structure-from-Motion stage
- `MVS:` — Multi-View Stereo stage
- `Subdivision pass N:` — Mesh subdivision stages
- `Texture coverage: X.X%` — Texture quality metric

### Verify output files
```bash
ls -la static/output/your_project_id_*
# Should show: .glb, .obj, _metadata.json
```

### Load metadata
```bash
python -c "import json; d=json.load(open('static/output/your_project_id_metadata.json')); [print(f'{k}: {v}') for k,v in d.items()]"
```

## Support

If issues persist:
1. Check the [ARCHITECTURE.md](ARCHITECTURE.md) for pipeline details
2. Open an issue on GitHub with:
   - Number of photos and their resolutions
   - Metadata JSON content
   - Full processing logs
   - Any error messages
