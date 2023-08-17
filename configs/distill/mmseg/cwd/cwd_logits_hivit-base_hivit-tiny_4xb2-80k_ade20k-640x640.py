
_base_ = [
    'mmseg::_base_/datasets/ade20k_640x640.py',
    'mmseg::_base_/schedules/schedule_80k.py',
    'mmseg::_base_/default_runtime.py'
]

teacher_ckpt = '../mmsegmentation/checkpoints/mae_hivit2_base_1600ep_ft100ep.pth'
teacher_cfg_path = 'mmseg::hivit2/hivit-base_upernet_1xb16-80k_ade-640x640.py'  # noqa: E501
student_cfg_path = 'mmseg::hivit2/hivit-tiny_upernet_1xb16-80k_ade-640x640.py'  # noqa: E501

model = dict(
    _scope_='mmrazor',
    type='FpnTeacherDistill',
    architecture=dict(cfg_path=student_cfg_path, pretrained=False),
    teacher=dict(cfg_path=teacher_cfg_path, pretrained=False),
    teacher_ckpt=teacher_ckpt,

    distiller=dict(
        type='ConfigurableDistiller',
        distill_losses=dict(
            loss_cwd_fpn1=dict(type='DimensionAdaptiveChannelWiseDivergence', tau=1, loss_weight=5, student_dim=96, teacher_dim=128),
            loss_cwd_fpn2=dict(type='DimensionAdaptiveChannelWiseDivergence', tau=1, loss_weight=5, student_dim=192, teacher_dim=256),
            loss_cwd_fpn3=dict(type='DimensionAdaptiveChannelWiseDivergence', tau=1, loss_weight=5, student_dim=384, teacher_dim=512),
        ),
        student_recorders=dict(
            logits=dict(type='ModuleOutputs', source='backbone'),
        ),
        teacher_recorders=dict(
            logits=dict(type='ModuleOutputs', source='backbone')
        ),
        loss_forward_mappings=dict(
            loss_cwd_fpn1=dict(
                preds_S=dict(from_student=True, recorder='logits', data_idx=0),
                preds_T=dict(from_student=False, recorder='logits', data_idx=0)
            ),
            loss_cwd_fpn2=dict(
                preds_S=dict(from_student=True, recorder='logits', data_idx=1),
                preds_T=dict(from_student=False, recorder='logits', data_idx=1)
            ),
            loss_cwd_fpn3=dict(
                preds_S=dict(from_student=True, recorder='logits', data_idx=2),
                preds_T=dict(from_student=False, recorder='logits', data_idx=2)
            ),
        )
    )
)

find_unused_parameters = True

val_cfg = dict(_delete_=True, type='mmrazor.SingleTeacherDistillValLoop')