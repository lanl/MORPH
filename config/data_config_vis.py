# src/utils/data_config.py
import os
        
class DataConfig(dict):
    def __init__(self, dataset_dir: str, project_root: str, split: str = None):
        mapping = {
            # ------------------- Pre-training sets ---------------------
            'DR2d_data_pdebench': {
                'file_path_dr2d':   os.path.join(dataset_dir, 'DR2d_data_pdebench', 
                                                 *( [split] if split else [] )),
                'file_path_dr2d_n': os.path.join(project_root, 'datasets', 'normalized_revin',
                                    'DR2d_data_pdebench', *( [split] if split else [] )),
            },
            'MHD3d_data_thewell': {
                'file_path_mhd3d':   os.path.join(dataset_dir, 'MHD3d_data_thewell', 
                                                  *( [split] if split else [] )),
                'file_path_mhd3d_n': os.path.join(project_root, 'datasets', 'normalized_revin',
                                     'MHD3d_data_thewell', *( [split] if split else [] )),
            },
            '1dcfd_pdebench': {
                'file_path_cfd1d':   os.path.join(dataset_dir, '1dcfd_pdebench', 
                                                  *( [split] if split else [] )),
                'file_path_cfd1d_n': os.path.join(project_root, 'datasets', 'normalized_revin',
                                     '1dcfd_pdebench', *( [split] if split else [] )),
            },
            '2dcfd_ic_pdebench': {
                'file_path_cfd2d_ic':   os.path.join(dataset_dir, '2dcfd_ic_pdebench', 
                                                     *( [split] if split else [] )),
                'file_path_cfd2d_ic_n': os.path.join(project_root, 'datasets', 'normalized_revin',
                                        '2dcfd_ic_pdebench', *( [split] if split else [] )),
            },
            '3dcfd_pdebench': {
                'file_path_cfd3d':   os.path.join(dataset_dir, '3dcfd_pdebench', 
                                                  *( [split] if split else [] )),
                'file_path_cfd3d_n': os.path.join(project_root, 'datasets', 'normalized_revin',
                                     '3dcfd_pdebench', *( [split] if split else [] )),
            },
            '2dSW_pdebench': {
                'file_path_sw2d':   os.path.join(dataset_dir, '2dSW_pdebench', 
                                                 *( [split] if split else [] )),
                'file_path_sw2d_n': os.path.join(project_root, 'datasets', 'normalized_revin',
                                    '2dSW_pdebench', *( [split] if split else [] )),
            },

            # ------------------- Finetuning sets ---------------------
            '1ddr_pdebench': {
                'file_path_dr1d': os.path.join(dataset_dir, '1ddr_pdebench',
                                               *( [split] if split else [] )),
                'file_path_dr1d_n': os.path.join(project_root, 'datasets', 'normalized_revin',
                                    '1ddr_pdebench', *( [split] if split else [] )),
                                },
            '2dcfd_pdebench': {
                'file_path_cfd2d': os.path.join(dataset_dir, '2dcfd_pdebench',
                                                *( [split] if split else [] )),
                'file_path_cfd2d_n': os.path.join(project_root, 'datasets', 'normalized_revin',
                                     '2dcfd_pdebench', *( [split] if split else [] ))
                                },
            '1dbe_pdebench':{
                'file_path_be1d':os.path.join(dataset_dir, '1dbe_pdebench',
                                              *( [split] if split else [] )),
                'file_path_be1d_n':os.path.join(project_root, 'datasets', 'normalized_revin',
                                   '1dbe_pdebench', *( [split] if split else [] ))
                            },
            '3dcfd_turb_pdebench':{
                'file_path_3dcfd_turb':os.path.join(dataset_dir, '3dcfd_turb_pdebench',
                                                    *( [split] if split else [] )),
                'file_path_3dcfd_turb_n':os.path.join(project_root, 'datasets', 'normalized_revin',
                                         '3dcfd_turb_pdebench', *( [split] if split else [] ))
                            },
            '2dgrayscottdr_thewell':{
                'file_path_2dgsdr':os.path.join(dataset_dir, '2dgrayscottdr_thewell',
                                                *( [split] if split else [] )),
                'file_path_2dgsdr_n':os.path.join(project_root, 'datasets', 'normalized_revin',
                                     '2dgrayscottdr_thewell', *( [split] if split else [] ))
                            },
            '3dturbgravitycool_thewell':{
                'file_path_3dtgc':os.path.join(dataset_dir, '3dturbgravitycool_thewell',
                                               *( [split] if split else [] )),
                'file_path_3dtgc_n':os.path.join(project_root, 'datasets', 'normalized_revin',
                                    '3dturbgravitycool_thewell', *( [split] if split else [] ))
                            },
            
            '2dFNS_KF_pdegym':{
                'file_path_2dfns_kf':os.path.join(dataset_dir, '2dFNS_KF_pdegym',
                                               *( [split] if split else [] )),
                'file_path_2dfns_kf_n':os.path.join(project_root, 'datasets', 'normalized_revin',
                                       '2dFNS_KF_pdegym', *( [split] if split else [] ))
                            },
            
            # ----------------------- New Pretraining sets -----------------------
            '2dCE_CRP_pdegym':{
                'file_path_2dce_crp':os.path.join(dataset_dir, '2dCE_CRP_pdegym',
                                               *( [split] if split else [] )),
                'file_path_2dce_crp_n':os.path.join(project_root, 'datasets', 'normalized_revin',
                                       '2dCE_CRP_pdegym', *( [split] if split else [] ))
                },
            
            '2dCE_RP_pdegym':{
                'file_path_2dce_rp':os.path.join(dataset_dir, '2dCE_RP_pdegym',
                                               *( [split] if split else [] )),
                'file_path_2dce_rp_n':os.path.join(project_root, 'datasets', 'normalized_revin',
                                       '2dCE_RP_pdegym', *( [split] if split else [] ))
                },
            
            '2dCE_KH_pdegym':{
                'file_path_2dce_kh':os.path.join(dataset_dir, '2dCE_KH_pdegym',
                                               *( [split] if split else [] )),
                'file_path_2dce_kh_n':os.path.join(project_root, 'datasets', 'normalized_revin',
                                       '2dCE_KH_pdegym', *( [split] if split else [] ))
                },
            
            '2dCE_Gauss_pdegym':{
                'file_path_2dce_gauss':os.path.join(dataset_dir, '2dCE_Gauss_pdegym',
                                               *( [split] if split else [] )),
                'file_path_2dce_gauss_n':os.path.join(project_root, 'datasets', 'normalized_revin',
                                       '2dCE_Gauss_pdegym', *( [split] if split else [] ))
                },
            
            '2dNS_Sines_pdegym':{
                'file_path_2dns_sines':os.path.join(dataset_dir, '2dNS_Sines_pdegym',
                                               *( [split] if split else [] )),
                'file_path_2dns_sines_n':os.path.join(project_root, 'datasets', 'normalized_revin',
                                       '2dNS_Sines_pdegym', *( [split] if split else [] ))
                },
            
            '2dNS_Gauss_pdegym':{
                'file_path_2dns_gauss':os.path.join(dataset_dir, '2dNS_Gauss_pdegym',
                                               *( [split] if split else [] )),
                'file_path_2dns_gauss_n':os.path.join(project_root, 'datasets', 'normalized_revin',
                                       '2dNS_Gauss_pdegym', *( [split] if split else [] ))
                }        
        }

        # initialize the dict with your mapping
        super().__init__(mapping)
