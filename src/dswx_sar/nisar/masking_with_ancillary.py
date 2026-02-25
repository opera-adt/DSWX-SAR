import copy
import logging
import mimetypes
import os
import time

import cv2
import numpy as np

from dswx_sar.common import (_dswx_sar_util,
                             _masking_with_ancillary,
                             _generate_log)
from dswx_sar.common._detect_inundated_vegetation import parse_ranges

from dswx_sar.nisar.dswx_ni_runconfig import (
    DSWX_NI_POL_DICT,
    _get_parser,
    RunConfig)


logger = logging.getLogger('dswx_sar')


def run(cfg):
    '''
    Remove the false water which have low backscattering based on the
    occurrence map and landcover map.
    '''
    logger.info('Starting DSWx-NI masking with ancillary data')

    t_all = time.time()
    outputdir = cfg.groups.product_path_group.scratch_path
    processing_cfg = cfg.groups.processing
    pol_list = copy.deepcopy(processing_cfg.polarizations)
    pol_options = processing_cfg.polarimetric_option

    if pol_options is not None:
        pol_list += pol_options

    pol_str = '_'.join(pol_list)
    co_pol_ind = next((i for i,p in enumerate(pol_list) 
                       if p in ('VV','HH')), 0)

    water_cfg = processing_cfg.reference_water
    ref_water_max = water_cfg.max_value

    # options for masking with ancillary data
    masking_ancillary_cfg = processing_cfg.masking_ancillary
    number_workers = masking_ancillary_cfg.number_cpu

    dry_water_area_threshold = masking_ancillary_cfg.water_threshold
    extended_landcover_flag = masking_ancillary_cfg.extended_darkland
    extend_minimum = masking_ancillary_cfg.extended_darkland_minimum_pixel
    water_buffer = masking_ancillary_cfg.extended_darkland_water_buffer
    hand_variation_mask = masking_ancillary_cfg.hand_variation_mask
    hand_variation_threshold = masking_ancillary_cfg.hand_variation_threshold
    landcover_masking_list = masking_ancillary_cfg.land_cover_darkland_list
    landcover_masking_extension_list = \
        masking_ancillary_cfg.land_cover_darkland_extension_list
    landcover_water_label = masking_ancillary_cfg.land_cover_water_label
    lines_per_block = masking_ancillary_cfg.line_per_block

    # Binary water map extracted from region growing
    water_map_tif_str = os.path.join(
        outputdir, f'region_growing_output_binary_{pol_str}.tif')
    water_map = _dswx_sar_util.read_geotiff(water_map_tif_str)
    water_meta = _dswx_sar_util.get_meta_from_tif(water_map_tif_str)

    # Filtered SAR intensity image
    filt_im_str = os.path.join(outputdir, f"filtered_image_{pol_str}.tif")

    # Reference water map
    interp_wbd_str = os.path.join(outputdir, 'interpolated_wbd.tif')
    interp_wbd = _dswx_sar_util.read_geotiff(interp_wbd_str)
    interp_wbd = np.array(interp_wbd, dtype='float32')

    # HAND
    hand_path_str = os.path.join(outputdir, 'interpolated_hand.tif')

    # Worldcover map
    landcover_path = os.path.join(outputdir, 'interpolated_landcover.tif')

    # GLAD
    glad_path = os.path.join(outputdir, 'interpolated_glad.tif')

    # Identify dark land candidate areas from landcover
    mask_obj = _masking_with_ancillary.FillMaskLandCover(landcover_path, 'WorldCover')
    mask_excluded_landcover = mask_obj.get_mask(
        mask_label=landcover_masking_list)
    del mask_obj

    glad_obj = _masking_with_ancillary.FillMaskLandCover(glad_path, 'GLAD')

    target_glad_class = ['1-24']
    inundated_vege_target = parse_ranges(target_glad_class)
    glad_mask_excluded_landcover = glad_obj.get_mask(
                mask_label=inundated_vege_target)
    mask_excluded_landcover_path = os.path.join(
        outputdir, 'initial_mask_excluded_landcover_ancillary.tif')

    _dswx_sar_util.save_dswx_product(
        mask_excluded_landcover | glad_mask_excluded_landcover,
        mask_excluded_landcover_path,
        geotransform=water_meta['geotransform'],
        projection=water_meta['projection'],
        scratch_dir=outputdir)

    # If `extended_landcover_flag`` is enabled, additional landcovers
    # spatially connected with `mask_excluded_landcover` are added.
    if extended_landcover_flag:
        logger.info('Extending landcover enabled.')
        initial_darkland_cand_path = \
            os.path.join(outputdir,
                        'initial_darkland_candidate_from_landcover_water_backscatter.tif')

        _masking_with_ancillary.get_darkland_from_intensity_ancillary(
            intensity_path=filt_im_str,
            landcover_path=mask_excluded_landcover_path,
            reference_water_path=interp_wbd_str,
            darkland_candidate_path=initial_darkland_cand_path,
            lines_per_block=lines_per_block,
            pol_list=pol_list,
            co_pol_threshold=masking_ancillary_cfg.co_pol_threshold,
            cross_pol_threshold=masking_ancillary_cfg.cross_pol_threshold,
            ref_water_max=ref_water_max,
            dry_water_area_threshold=dry_water_area_threshold)
        mask_excluded_landcover_original = copy.deepcopy(mask_excluded_landcover)

        potential_water_glad_class = ['100-230']
        potential_water_glad_class_whole = parse_ranges(potential_water_glad_class)
        potential_water_glad = glad_obj.get_mask(mask_label=potential_water_glad_class_whole)
        # region growing does not cover the water areas
        initial_mask_excluded_landcover = _dswx_sar_util.read_geotiff(initial_darkland_cand_path)
        rg_excluded_area = np.logical_or(interp_wbd > dry_water_area_threshold,
                                         potential_water_glad)

        mask_excluded_landcover = _masking_with_ancillary.extend_land_cover_v2(
            landcover_path=landcover_path,
            reference_landcover_binary=initial_mask_excluded_landcover,
            target_landcover=landcover_masking_extension_list,
            water_landcover=landcover_water_label,
            exclude_area_rg=rg_excluded_area,
            water_buffer=water_buffer,
            minimum_pixel=extend_minimum,
            metainfo=water_meta,
            scratch_dir=outputdir)
        logger.info('Landcover extension completed.')
        mask_excluded_landcover = np.logical_or(mask_excluded_landcover,
                                                mask_excluded_landcover_original)
        mask_excluded_landcover_path = os.path.join(
            outputdir, 'mask_excluded_landcover_ancillary.tif')
        _dswx_sar_util.save_dswx_product(
            mask_excluded_landcover,
            mask_excluded_landcover_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            scratch_dir=outputdir)

    # 2) Create initial mask binary
    # mask_excluded indicates the areas satisfying all conditions which are
    # 1: `dry_water_area_threshold` of water occurrence over 37 year (Pekel)
    # 2: specified landcovers (bare ground, sparse vegetation, urban, moss...)
    # 3: low backscattering area
    # mask = intersect of landcover mask + no water + dark land

    darkland_cand_path = \
        os.path.join(outputdir,
                     'darkland_candidate_from_landcover_water_backscatter.tif')

    _masking_with_ancillary.get_darkland_from_intensity_ancillary(
        intensity_path=filt_im_str,
        landcover_path=mask_excluded_landcover_path,
        reference_water_path=interp_wbd_str,
        darkland_candidate_path=darkland_cand_path,
        lines_per_block=lines_per_block,
        pol_list=pol_list,
        co_pol_threshold=masking_ancillary_cfg.co_pol_threshold,
        cross_pol_threshold=masking_ancillary_cfg.cross_pol_threshold,
        ref_water_max=ref_water_max,
        dry_water_area_threshold=dry_water_area_threshold)

    # 3) The water candidates extracted in the previous step (region growing)
    # can have dark land and water in one singe polygon where dark land and
    # water are spatially connected. Here 'split_extended_water' checks
    # if the backscatter splits into smaller pieces that are distinguishable.
    # So 'split_extended_water' checks for bimodality for each polygon.
    # If so, the code calculates a threshold and checks the bimodality
    # for each split area.
    input_map = np.where(water_map == 1, 1, 0)
    del water_map

    _dswx_sar_util.save_dswx_product(
        input_map,
        os.path.join(outputdir, 'input_image_test.tif'),
        geotransform=water_meta['geotransform'],
        projection=water_meta['projection'],
        scratch_dir=outputdir)
    del input_map

    input_file_dict = {'intensity': filt_im_str,
                       'landcover': landcover_path,
                       'reference_water': interp_wbd_str,
                       'water_mask': water_map_tif_str}

    split_mask_water_path = \
        os.path.join(outputdir, 'split_mask_water_masking.tif')
    _masking_with_ancillary.split_extended_water_parallel_v2(
        water_mask_path=os.path.join(outputdir, 'input_image_test.tif'),
        output_path=split_mask_water_path,
        pol_ind=co_pol_ind,
        outputdir=outputdir,
        input_dict=input_file_dict,
        number_workers=number_workers,
        input_lines_per_block=lines_per_block)

    # 4) re-define false water candidate estimated from
    # 'split_extended_water_parallel'
    # by considering the spatial coverage with the ancillary files
    false_water_candidate_path = os.path.join(
        outputdir, 'intermediate_false_water_candidate.tif')
    _dswx_sar_util.merge_binary_layers(
        layer_list=[split_mask_water_path,
                    water_map_tif_str],
        value_list=[0, 1],
        merged_layer_path=false_water_candidate_path,
        lines_per_block=lines_per_block,
        mode='and',
        cog_flag=True,
        scratch_dir=outputdir)

    adjacent_false_positive_binary_path = \
        os.path.join(outputdir, 'false_positive_connected_water.tif')

    _masking_with_ancillary.compute_spatial_coverage_from_ancillary_parallel(
            false_water_binary_path=false_water_candidate_path,
            reference_water_path=interp_wbd_str,
            mask_landcover_path=mask_excluded_landcover_path,
            output_file_path=adjacent_false_positive_binary_path,
            outputdir=outputdir,
            water_max_value=ref_water_max,
            number_workers=number_workers,
            lines_per_block=lines_per_block)

    if hand_variation_mask:
        darkland_removed_path = \
            os.path.join(outputdir, 'masking_ancillary_darkland_removed.tif')
        water_tif_str = \
            os.path.join(outputdir, f"refine_landcover_binary_{pol_str}.tif")
    else:
        darkland_removed_path = \
             os.path.join(outputdir, f"refine_landcover_binary_{pol_str}.tif")

    _dswx_sar_util.merge_binary_layers(
        layer_list=[adjacent_false_positive_binary_path,
                    darkland_cand_path,
                    water_map_tif_str],
        value_list=[0, 0, 1],
        merged_layer_path=darkland_removed_path,
        lines_per_block=lines_per_block,
        mode='and',
        cog_flag=True,
        scratch_dir=outputdir)

    orig = _dswx_sar_util.read_geotiff(water_map_tif_str).astype(bool)
    final = _dswx_sar_util.read_geotiff(darkland_removed_path).astype(bool)
    removed = np.count_nonzero(orig & ~final)
    kept = np.count_nonzero(final)
    logger.info(f"[masking] kept={kept:,} removed={removed:,} removed_pct={removed/(kept+removed+1e-9):.3%}")

    if hand_variation_mask:
        _masking_with_ancillary.hand_filter_along_boundary(
            target_area_path=darkland_removed_path,
            height_std_threshold=hand_variation_threshold,
            hand_path=hand_path_str,
            output_path=water_tif_str,
            debug_mode=processing_cfg.debug_mode,
            metainfo=water_meta,
            scratch_dir=outputdir)

    if processing_cfg.debug_mode:
        excluded_path = os.path.join(
            outputdir, f"landcover_exclude_{pol_str}.tif")
        _dswx_sar_util.save_dswx_product(
            mask_excluded_landcover,
            excluded_path,
            geotransform=water_meta['geotransform'],
            projection=water_meta['projection'],
            description='Water classification (WTR)',
            scratch_dir=outputdir
            )

    t_all_elapsed = time.time() - t_all
    logger.info("successfully ran landcover masking in "
                f"{t_all_elapsed:.3f} seconds")


def main():

    parser = _get_parser()

    args = parser.parse_args()

    mimetypes.add_type("text/yaml", ".yaml", strict=True)
    flag_first_file_is_text = 'text' in mimetypes.guess_type(
        args.input_yaml[0])[0]

    if len(args.input_yaml) > 1 and flag_first_file_is_text:
        logger.info('ERROR only one runconfig file is allowed')
        return

    if flag_first_file_is_text:
        cfg = RunConfig.load_from_yaml(args.input_yaml[0], 'dswx_ni', args)

    _generate_log.configure_log_file(cfg.groups.log_file)

    processing_cfg = cfg.groups.processing
    pol_mode = processing_cfg.polarization_mode
    pol_list = processing_cfg.polarizations
    if pol_mode == 'MIX_DUAL_POL':
        proc_pol_set = [DSWX_NI_POL_DICT['DV_POL'],
                        DSWX_NI_POL_DICT['DH_POL']]
    elif pol_mode == 'MIX_SINGLE_POL':
        proc_pol_set = [DSWX_NI_POL_DICT['SV_POL'],
                        DSWX_NI_POL_DICT['SH_POL']]
    else:
        proc_pol_set = [pol_list]

    for pol_set in proc_pol_set:
        processing_cfg.polarizations = pol_set
        run(cfg)


if __name__ == '__main__':
    main()
