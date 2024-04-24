import copy
import logging
import mimetypes
import numpy as np
import os
import pathlib
import time

from pathlib import Path

from dswx_sar import dswx_sar_util, filter_SAR, generate_log, pre_processing
from dswx_sar.dswx_ni_runconfig import (DSWX_S1_POL_DICT,
                                        _get_parser,
                                        RunConfig)
from dswx_sar.dswx_sar_util import check_gdal_raster_s3
from dswx_sar.masking_with_ancillary import get_label_landcover_esa_10


logger = logging.getLogger('dswx_sar')


def run(cfg):

    logger.info("")
    logger.info('Starting DSWx-S1 Preprocessing')

    t_all = time.time()
    processing_cfg = cfg.groups.processing
    dynamic_data_cfg = cfg.groups.dynamic_ancillary_file_group

    input_list = cfg.groups.input_file_group.input_file_path
    scratch_dir = cfg.groups.product_path_group.scratch_path

    wbd_file = dynamic_data_cfg.reference_water_file
    landcover_file = dynamic_data_cfg.worldcover_file
    dem_file = dynamic_data_cfg.dem_file
    hand_file = dynamic_data_cfg.hand_file

    ref_water_max = processing_cfg.reference_water.max_value
    ref_water_no_data = processing_cfg.reference_water.no_data_value
    pol_list = copy.deepcopy(processing_cfg.polarizations)
    pol_options = processing_cfg.polarimetric_option

    if pol_options is not None:
        pol_list += pol_options

    pol_all_str = '_'.join(pol_list)
    co_pol = processing_cfg.copol
    cross_pol = processing_cfg.crosspol

    filter_size = processing_cfg.filter.window_size
    filter_flag = processing_cfg.filter.enabled
    line_per_block = processing_cfg.filter.line_per_block

    mosaic_prefix = processing_cfg.mosaic.mosaic_prefix
    if mosaic_prefix is None:
        mosaic_prefix = 'mosaic'

    # configure if input is single/multi directory/file
    num_input_path = len(input_list)

    if num_input_path == 1:
        logger.info('Single input RTC is found.')
        mosaic_flag = False
    else:
        mosaic_flag = True
        logger.info(f'{num_input_path} input RTC is found.')
    ref_filename = f'{scratch_dir}/{mosaic_prefix}_{pol_list[0]}.tif'
    ref_filename = f'{ref_filename[:]}'

    logger.info(f'Ancillary data is reprojected using {ref_filename}')

    pathlib.Path(scratch_dir).mkdir(parents=True, exist_ok=True)
    filtered_images_str = f"filtered_image_{pol_all_str}.tif"
    filtered_image_path = os.path.join(
        scratch_dir, filtered_images_str)

    # read metadata from Geotiff File.
    im_meta = dswx_sar_util.get_meta_from_tif(ref_filename)

    # create instance to relocate ancillary data
    ancillary_reloc = pre_processing.AncillaryRelocation(
        ref_filename,
        scratch_dir)

    # Note : landcover should precede before reference water.
    relocated_ancillary_filename_set = {
        'dem': 'interpolated_DEM.tif',
        'hand': 'interpolated_hand.tif',
        'landcover': 'interpolated_landcover.tif',
        'reference_water': 'interpolated_wbd.tif',
    }

    input_ancillary_filename_set = {
        'dem': dem_file,
        'hand': hand_file,
        'landcover': landcover_file,
        'reference_water': wbd_file,
    }

    landcover_label = get_label_landcover_esa_10()

    for anc_type, anc_filename in relocated_ancillary_filename_set.items():
        input_anc_path = input_ancillary_filename_set[anc_type]
        ancillary_path = Path(
            os.path.join(scratch_dir, anc_filename))

        # Check if input ancillary data is valid.
        if not os.path.isfile(input_anc_path) and \
           not check_gdal_raster_s3(input_anc_path, raise_error=False):
            err_msg = f'Input {anc_type} file not found'
            raise FileNotFoundError(err_msg)

        if anc_type in ['reference_water']:
            no_data = ref_water_no_data
        elif anc_type in ['landcover']:
            no_data = landcover_label['No_data']
        else:
            no_data = np.nan

        # crop or relocate ancillary images to fit the reference
        # intensity (RTC) image.
        if not ancillary_path.is_file():
            logger.info(f'Relocated {anc_type} file will be created from '
                        f'{input_anc_path}.')
            ancillary_reloc.relocate(
                input_anc_path,
                anc_filename,
                method='near',
                no_data=no_data)

        # check if relocated ancillaries are filled with invalid values
        validation_result = pre_processing.validate_gtiff(
            os.path.join(scratch_dir, anc_filename),
            [no_data, np.inf])

        if validation_result not in ['okay']:
            if (anc_type in ['dem', 'hand']) and \
               (validation_result in ['nan_value', 'invalid_only']):
                err_msg = f'Unable to get valid {anc_type}'
                raise ValueError(err_msg)

            elif anc_type in ['landcover']:
                logger.warning(f'{anc_type} has invalid values.')

            elif anc_type in ['reference_water']:
                logger.warning(f'{anc_type} has invalid values.')
                # update reference water if any values of reference water
                # are invalid.
                pre_processing.replace_reference_water_nodata_from_ancillary(
                    os.path.join(scratch_dir,
                                 relocated_ancillary_filename_set[
                                     'reference_water']),
                    os.path.join(scratch_dir,
                                 relocated_ancillary_filename_set[
                                     'landcover']),
                    os.path.join(scratch_dir,
                                 relocated_ancillary_filename_set[
                                     'hand']),
                    reference_water_max=ref_water_max,
                    reference_water_no_data=ref_water_no_data,
                    line_per_block=line_per_block
                )

    # apply SAR filtering
    pad_shape = (filter_size, 0)
    block_params = dswx_sar_util.block_param_generator(
        lines_per_block=line_per_block,
        data_shape=(im_meta['length'],
                    im_meta['width']),
        pad_shape=pad_shape)

    for block_ind, block_param in enumerate(block_params):
        output_image_set = []
        for polind, pol in enumerate(pol_list):
            logger.info(f'  block processing {block_ind} - {pol}')

            if pol in ['ratio', 'span']:

                # If ratio/span is in the list,
                # then compute the ratio from VVVV and VHVH
                temp_pol_list = co_pol + cross_pol
                logger.info(f'  >> computing {pol} {temp_pol_list}')

                temp_raster_set = []
                for temp_pol in temp_pol_list:
                    filename = \
                        f'{scratch_dir}/{mosaic_prefix}_{temp_pol}.tif'

                    block_data = dswx_sar_util.get_raster_block(
                        filename,
                        block_param)

                    temp_raster_set.append(block_data)

                temp_raster_set = np.array(temp_raster_set)
                if pol in ['ratio']:
                    ratio = pre_processing.pol_ratio(
                        np.squeeze(temp_raster_set[0, :, :]),
                        np.squeeze(temp_raster_set[1, :, :]))
                    output_image_set.append(ratio)
                    logger.info(f'  computing ratio {co_pol}/{cross_pol}')

                if pol in ['span']:
                    span = np.squeeze(temp_raster_set[0, :, :] +
                                      2 * temp_raster_set[1, :, :])
                    output_image_set.append(span)
            else:
                intensity_path = \
                    f'{scratch_dir}/{mosaic_prefix}_{pol}.tif'

                intensity = dswx_sar_util.get_raster_block(
                    intensity_path, block_param)
                # need to replace 0 value in padded area to NaN.
                intensity[intensity == 0] = np.nan
                if filter_flag:
                    filtered_intensity = filter_SAR.lee_enhanced_filter(
                                    intensity,
                                    win_size=filter_size)
                else:
                    filtered_intensity = intensity

                output_image_set.append(filtered_intensity)

        output_image_set = np.array(output_image_set, dtype='float32')
        output_image_set[output_image_set == 0] = np.nan

        dswx_sar_util.write_raster_block(
            out_raster=filtered_image_path,
            data=output_image_set,
            block_param=block_param,
            geotransform=im_meta['geotransform'],
            projection=im_meta['projection'],
            datatype='float32')

    dswx_sar_util._save_as_cog(filtered_image_path, scratch_dir)

    no_data_geotiff_path = os.path.join(
        scratch_dir, f"no_data_area_{pol_all_str}.tif")

    dswx_sar_util.get_invalid_area(
        os.path.join(scratch_dir, filtered_images_str),
        no_data_geotiff_path,
        scratch_dir=scratch_dir,
        lines_per_block=line_per_block)

    if processing_cfg.debug_mode:
        filtered_intensity = dswx_sar_util.read_geotiff(filtered_image_path)
        if filtered_intensity.ndim == 2:
            filtered_intensity = np.expand_dims(filtered_intensity,
                                                axis=0)
        for pol_ind, pol in enumerate(pol_list):
            if pol == 'ratio':
                immin, immax = None, None
            else:
                immin, immax = -30, 0
            single_intensity = np.squeeze(filtered_intensity[pol_ind, :, :])
            dswx_sar_util.intensity_display(
                single_intensity,
                scratch_dir,
                pol,
                immin,
                immax)

            if pol in ['ratio']:
                dswx_sar_util.save_raster_gdal(
                    data=single_intensity,
                    output_file=os.path.join(
                        scratch_dir, f'intensity_{pol}.tif'),
                    geotransform=im_meta['geotransform'],
                    projection=im_meta['projection'],
                    scratch_dir=scratch_dir)
            else:
                dswx_sar_util.save_raster_gdal(
                    data=10 * np.log10(single_intensity),
                    output_file=os.path.join(
                        scratch_dir, f'intensity_{pol}_db.tif'),
                    geotransform=im_meta['geotransform'],
                    projection=im_meta['projection'],
                    scratch_dir=scratch_dir)

    t_all_elapsed = time.time() - t_all
    logger.info("successfully ran pre-processing in "
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

    generate_log.configure_log_file(cfg.groups.log_file)

    processing_cfg = cfg.groups.processing
    pol_mode = processing_cfg.polarization_mode
    pol_list = processing_cfg.polarizations
    if pol_mode == 'MIX_DUAL_POL':
        proc_pol_set = [DSWX_S1_POL_DICT['DV_POL'],
                        DSWX_S1_POL_DICT['DH_POL']]
    elif pol_mode == 'MIX_SINGLE_POL':
        proc_pol_set = [DSWX_S1_POL_DICT['SV_POL'],
                        DSWX_S1_POL_DICT['SH_POL']]
    else:
        proc_pol_set = [pol_list]

    for pol_set in proc_pol_set:
        processing_cfg.polarizations = pol_set
        run(cfg)


if __name__ == '__main__':
    main()
