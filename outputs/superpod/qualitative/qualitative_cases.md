# Qualitative cases (LIBERO val)

## Successes

### idx=0
- task: `pick up the butter and place it in the basket`
- gold: `moved=yes; lifted=no; gripper=open; progress=flat; success_near=no`
- pred: `moved=yes; lifted=no; gripper=open; progress=flat; success_near=no`
- image: `outputs/superpod/qualitative/frames/ok_00.jpg`

### idx=1
- task: `pick up the butter and place it in the basket`
- gold: `moved=yes; lifted=no; gripper=open; progress=flat; success_near=no`
- pred: `moved=yes; lifted=no; gripper=open; progress=flat; success_near=no`
- image: `outputs/superpod/qualitative/frames/ok_01.jpg`

### idx=2
- task: `pick up the butter and place it in the basket`
- gold: `moved=yes; lifted=no; gripper=open; progress=down; success_near=no`
- pred: `moved=yes; lifted=no; gripper=open; progress=down; success_near=no`
- image: `outputs/superpod/qualitative/frames/ok_02.jpg`

### idx=3
- task: `pick up the butter and place it in the basket`
- gold: `moved=yes; lifted=no; gripper=open; progress=down; success_near=no`
- pred: `moved=yes; lifted=no; gripper=open; progress=down; success_near=no`
- image: `outputs/superpod/qualitative/frames/ok_03.jpg`

### idx=4
- task: `pick up the butter and place it in the basket`
- gold: `moved=yes; lifted=no; gripper=open; progress=down; success_near=no`
- pred: `moved=yes; lifted=no; gripper=open; progress=down; success_near=no`
- image: `outputs/superpod/qualitative/frames/ok_04.jpg`

## Failures

### idx=24
- task: `pick up the butter and place it in the basket`
- gold: `moved=yes; lifted=no; gripper=close; progress=up; success_near=no`
- pred: `moved=yes; lifted=no; gripper=close; progress=down; success_near=no`
- image: `outputs/superpod/qualitative/frames/fail_00.jpg`

### idx=25
- task: `pick up the butter and place it in the basket`
- gold: `moved=yes; lifted=no; gripper=close; progress=up; success_near=no`
- pred: `moved=yes; lifted=no; gripper=close; progress=flat; success_near=no`
- image: `outputs/superpod/qualitative/frames/fail_01.jpg`

### idx=30
- task: `pick up the butter and place it in the basket`
- gold: `moved=yes; lifted=no; gripper=open; progress=flat; success_near=no`
- pred: `moved=yes; lifted=no; gripper=open; progress=flat; success_near=yes`
- image: `outputs/superpod/qualitative/frames/fail_02.jpg`

### idx=41
- task: `pick up the cream cheese and place it in the basket`
- gold: `moved=yes; lifted=no; gripper=open; progress=down; success_near=no`
- pred: `moved=yes; lifted=no; gripper=open; progress=flat; success_near=no`
- image: `outputs/superpod/qualitative/frames/fail_03.jpg`

### idx=54
- task: `pick up the cream cheese and place it in the basket`
- gold: `moved=yes; lifted=no; gripper=close; progress=flat; success_near=yes`
- pred: `moved=yes; lifted=no; gripper=close; progress=flat; success_near=no`
- image: `outputs/superpod/qualitative/frames/fail_04.jpg`
