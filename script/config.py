args = dict(
    train=dict(
        versions=['v1', 'v2', 'v3', 'v4', 'v5'],
        model_type="rf",
        save_dir="/path/to/data/AI_osteoporosis/result/equal_wholeside/rf",
        test_ratio=0.2,
        batch_size=512,
        num_df=256,
        epochs=2000,
        random_state=2021,
        texture_tr=0.01,
        deep_tr=0.03,
        same_feature_each_side=True,
        deep_feature_dir="/path/to/data/AI_osteoporosis/df_wholeside",
        oneside=False,
        texture_feature_dir="/path/to/data/AI_osteoporosis/texture_feature",
        snuh_brmh_clinic_feature_file="/path/to/data/AI_osteoporosis/snuh_brmh_clinic_total.pickle",
        label_file="/path/to/data/AI_osteoporosis/label_dict_final.pickle",
        data_type=["development"],
        description="Development",
    ),

    test=dict(
        versions=['v1', 'v2', 'v3', 'v4', 'v5'],
        model_type="rf",
        model_path="/path/to/data/AI_osteoporosis/result/cropped/rf",
        save_dir="/path/to/data/AI_osteoporosis/result/cropped/result/rf",
        batch_size=256,
        oneside=False,
        same_feature_each_side=False,
        num_df=256,
        deep_feature_dir="/path/to/data/AI_osteoporosis/df_cropped",
        texture_feature_dir="/path/to/data/AI_osteoporosis/texture_feature",
        snuh_clinic_feature_file="/path/to/data/AI_osteoporosis/whole_clinic_final.pickle",
        brmh_clinic_feature_file="/path/to/data/AI_osteoporosis/brmh_clinic_final_ver2.pickle",
        snuh_brmh_clinic_feature_file="/path/to/data/AI_osteoporosis/snuh_brmh_clinic_total.pickle",
        label_file="/path/to/data/AI_osteoporosis/label_dict_final.pickle",
        data_type=["brmh", "test", "valid"],
        description="Test",
    ),
)

if __name__ == '__main__':
    pass
