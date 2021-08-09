import torch
import torch.nn.functional as F

from sutils import diff_operators
import modules


def image_mse(mask, model_output, gt):
    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean()}


def image_l1(mask, model_output, gt):
    if mask is None:
        return {'img_loss': torch.abs(model_output['model_out'] - gt['img']).mean()}
    else:
        return {'img_loss': (mask * torch.abs(model_output['model_out'] - gt['img'])).mean()}


def image_mse_TV_prior(mask, k1, model, model_output, gt):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(diff_operators.gradient(
                    rand_output['model_out'], rand_output['model_in']))).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(diff_operators.gradient(
                    rand_output['model_out'], rand_output['model_in']))).mean()}


def image_mse_FH_prior(mask, k1, model, model_output, gt):
    coords_rand = 2 * (torch.rand((model_output['model_in'].shape[0],
                                   model_output['model_in'].shape[1] // 2,
                                   model_output['model_in'].shape[2])).cuda() - 0.5)
    rand_input = {'coords': coords_rand}
    rand_output = model(rand_input)

    img_hessian, status = diff_operators.hessian(rand_output['model_out'],
                                                 rand_output['model_in'])
    img_hessian = img_hessian.view(*img_hessian.shape[0:2], -1)
    hessian_norm = img_hessian.norm(dim=-1, keepdim=True)

    if mask is None:
        return {'img_loss': ((model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(hessian_norm)).mean()}
    else:
        return {'img_loss': (mask * (model_output['model_out'] - gt['img']) ** 2).mean(),
                'prior_loss': k1 * (torch.abs(hessian_norm)).mean()}


def latent_loss(model_output):
    return torch.mean(model_output['latent_vec'] ** 2)


def hypo_weight_loss(model_output):
    weight_sum = 0
    total_weights = 0

    for weight in model_output['hypo_params'].values():
        weight_sum += torch.sum(weight ** 2)
        total_weights += weight.numel()

    return weight_sum * (1 / total_weights)


def image_hypernetwork_loss(mask, kl, fw, model_output, gt):
    return {'img_loss': image_mse(mask, model_output, gt)['img_loss'],
            'latent_loss': kl * latent_loss(model_output),
            'hypo_weight_loss': fw * hypo_weight_loss(model_output)}


def function_mse(model_output, gt):
    return {'func_loss': ((model_output['model_out'] - gt['func']) ** 2).mean()}


def gradients_mse(model_output, gt):
    # compute gradients on the model
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt['gradients']).pow(2).sum(-1))
    return {'gradients_loss': gradients_loss}


def gradients_color_mse(model_output, gt):
    # compute gradients on the model
    gradients_r = diff_operators.gradient(model_output['model_out'][..., 0], model_output['model_in'])
    gradients_g = diff_operators.gradient(model_output['model_out'][..., 1], model_output['model_in'])
    gradients_b = diff_operators.gradient(model_output['model_out'][..., 2], model_output['model_in'])
    gradients = torch.cat((gradients_r, gradients_g, gradients_b), dim=-1)
    # compare them with the ground-truth
    weights = torch.tensor([1e1, 1e1, 1., 1., 1e1, 1e1]).cuda()
    gradients_loss = torch.mean((weights * (gradients[0:2] - gt['gradients']).pow(2)).sum(-1))
    return {'gradients_loss': gradients_loss}


def laplace_mse(model_output, gt):
    # compute laplacian on the model
    laplace = diff_operators.laplace(model_output['model_out'], model_output['model_in'])
    # compare them with the ground truth
    laplace_loss = torch.mean((laplace - gt['laplace']) ** 2)
    return {'laplace_loss': laplace_loss}


def wave_pml(model_output, gt):
    source_boundary_values = gt['source_boundary_values']
    x = model_output['model_in']  # (meta_batch_size, num_points, 3)
    y = model_output['model_out']  # (meta_batch_size, num_points, 1)
    squared_slowness = gt['squared_slowness']
    dirichlet_mask = gt['dirichlet_mask']
    batch_size = x.shape[1]

    du, status = diff_operators.jacobian(y, x)
    dudt = du[..., 0]

    if torch.all(dirichlet_mask):
        diff_constraint_hom = torch.Tensor([0])
    else:
        hess, status = diff_operators.jacobian(du[..., 0, :], x)
        lap = hess[..., 1, 1, None] + hess[..., 2, 2, None]
        dudt2 = hess[..., 0, 0, None]
        diff_constraint_hom = dudt2 - 1 / squared_slowness * lap

    dirichlet = y[dirichlet_mask] - source_boundary_values[dirichlet_mask]
    neumann = dudt[dirichlet_mask]

    return {'dirichlet': torch.abs(dirichlet).sum() * batch_size / 1e1,
            'neumann': torch.abs(neumann).sum() * batch_size / 1e2,
            'diff_constraint_hom': torch.abs(diff_constraint_hom).sum()}


def helmholtz_pml(model_output, gt):
    source_boundary_values = gt['source_boundary_values']

    if 'rec_boundary_values' in gt:
        rec_boundary_values = gt['rec_boundary_values']

    wavenumber = gt['wavenumber'].float()
    x = model_output['model_in']  # (meta_batch_size, num_points, 2)
    y = model_output['model_out']  # (meta_batch_size, num_points, 2)
    squared_slowness = gt['squared_slowness'].repeat(1, 1, y.shape[-1] // 2)
    batch_size = x.shape[1]

    full_waveform_inversion = False
    if 'pretrain' in gt:
        pred_squared_slowness = y[:, :, -1] + 1.
        if torch.all(gt['pretrain'] == -1):
            full_waveform_inversion = True
            pred_squared_slowness = torch.clamp(y[:, :, -1], min=-0.999) + 1.
            squared_slowness_init = torch.stack((torch.ones_like(pred_squared_slowness),
                                                 torch.zeros_like(pred_squared_slowness)), dim=-1)
            squared_slowness = torch.stack((pred_squared_slowness, torch.zeros_like(pred_squared_slowness)), dim=-1)
            squared_slowness = torch.where((torch.abs(x[..., 0, None]) > 0.75) | (torch.abs(x[..., 1, None]) > 0.75),
                                           squared_slowness_init, squared_slowness)

        y = y[:, :, :-1]

    du, status = diff_operators.jacobian(y, x)
    dudx1 = du[..., 0]
    dudx2 = du[..., 1]

    a0 = 5.0

    # let pml extend from -1. to -1 + Lpml and 1 - Lpml to 1.0
    Lpml = 0.5
    dist_west = -torch.clamp(x[..., 0] + (1.0 - Lpml), max=0)
    dist_east = torch.clamp(x[..., 0] - (1.0 - Lpml), min=0)
    dist_south = -torch.clamp(x[..., 1] + (1.0 - Lpml), max=0)
    dist_north = torch.clamp(x[..., 1] - (1.0 - Lpml), min=0)

    sx = wavenumber * a0 * ((dist_west / Lpml) ** 2 + (dist_east / Lpml) ** 2)[..., None]
    sy = wavenumber * a0 * ((dist_north / Lpml) ** 2 + (dist_south / Lpml) ** 2)[..., None]

    ex = torch.cat((torch.ones_like(sx), -sx / wavenumber), dim=-1)
    ey = torch.cat((torch.ones_like(sy), -sy / wavenumber), dim=-1)

    A = modules.compl_div(ey, ex).repeat(1, 1, dudx1.shape[-1] // 2)
    B = modules.compl_div(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)
    C = modules.compl_mul(ex, ey).repeat(1, 1, dudx1.shape[-1] // 2)

    a, _ = diff_operators.jacobian(modules.compl_mul(A, dudx1), x)
    b, _ = diff_operators.jacobian(modules.compl_mul(B, dudx2), x)

    a = a[..., 0]
    b = b[..., 1]
    c = modules.compl_mul(modules.compl_mul(C, squared_slowness), wavenumber ** 2 * y)

    diff_constraint_hom = a + b + c
    diff_constraint_on = torch.where(source_boundary_values != 0.,
                                     diff_constraint_hom - source_boundary_values,
                                     torch.zeros_like(diff_constraint_hom))
    diff_constraint_off = torch.where(source_boundary_values == 0.,
                                      diff_constraint_hom,
                                      torch.zeros_like(diff_constraint_hom))
    if full_waveform_inversion:
        data_term = torch.where(rec_boundary_values != 0, y - rec_boundary_values, torch.Tensor([0.]).cuda())
    else:
        data_term = torch.Tensor([0.])

        if 'pretrain' in gt:  # we are not trying to solve for velocity
            data_term = pred_squared_slowness - squared_slowness[..., 0]

    return {'diff_constraint_on': torch.abs(diff_constraint_on).sum() * batch_size / 1e3,
            'diff_constraint_off': torch.abs(diff_constraint_off).sum(),
            'data_term': torch.abs(data_term).sum() * batch_size / 1}


def sdf(model_output, gt):
    '''
       x: batch of input coordinates
       y: usually the output of the trial_soln function
       '''
    gt_sdf = gt['sdf']
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    gradient = diff_operators.gradient(pred_sdf, coords)

    # Wherever boundary_values is not equal to zero, we interpret it as a boundary constraint.
    sdf_constraint = torch.where(gt_sdf != -1, pred_sdf, torch.zeros_like(pred_sdf))
    inter_constraint = torch.where(gt_sdf != -1, torch.zeros_like(pred_sdf), torch.exp(-1e2 * torch.abs(pred_sdf)))
    normal_constraint = torch.where(gt_sdf != -1, 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None],
                                    torch.zeros_like(gradient[..., :1]))
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)
    # Exp      # Lapl
    # -----------------
    return {'sdf': torch.abs(sdf_constraint).mean() * 3e3,  # 1e4      # 3e3
            'inter': inter_constraint.mean() * 1e2,  # 1e2                   # 1e3
            'normal_constraint': normal_constraint.mean() * 1e2,  # 1e2
            'grad_constraint': grad_constraint.mean() * 5e1}  # 1e1      # 5e1

# inter = 3e3 for ReLU-PE


def depth_approx(model_output, gt):
    """
    x: batch of input coordinates
    y: depth output
    """

    gt_depth = gt['depth']
    pred_depth = model_output['model_out']
    residue = pred_depth - gt_depth
    mask = torch.logical_not(torch.isinf(gt_depth))
    residue = torch.where(mask, residue, torch.zeros_like(pred_depth))
    # Exp      # Lapl
    # -----------------
    return {'depth_mse': (residue ** 2).mean()}  # 1e1      # 5e1


def render_NL_img_mse(mask, model_output, gt):
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    dx, dy = gradients[:,:,0], gradients[:,:,1]

    xx, yy =model_output['model_in'][:,:,0], model_output['model_in'][:,:,1]
    zz = model_output['model_out']
    nx = - dx.unsqueeze(2)
    ny = - dy.unsqueeze(2)
    nz = torch.ones_like(nx)
    normal_set = torch.stack([nx, ny, nz], dim=2).squeeze(3)
    N_norm = torch.norm(normal_set, p=2, dim=2)
    normal_dir = normal_set / N_norm.unsqueeze(2)

    point_set = torch.stack([xx.unsqueeze(2), yy.unsqueeze(2), zz], dim=2).squeeze(3)

    # now we test use the rendering error for all image sequence
    batch_size, numLEDs, _ =  gt['LED_loc'].shape
    img_loss_all = 0
    for i in range(numLEDs):
        LED_loc = gt['LED_loc'][:, i].unsqueeze(1)
        lights = LED_loc - point_set
        L_norm = torch.norm(lights, p=2, dim=2).unsqueeze(2)
        light_dir = lights / L_norm
        light_falloff = torch.pow(L_norm, -2)

        shading =  torch.sum(light_dir * normal_dir, dim=2, keepdims=True)
        attach_shadow = torch.nn.ReLU()
        img = light_falloff * attach_shadow(shading)
        img_loss = (mask * ((img - gt['img'][:, :, i].unsqueeze(2)) ** 2)).mean()
        img_loss_all =  img_loss_all + img_loss

    normal_loss = 1 - F.cosine_similarity(normal_dir, gt['normal_gt'], dim=-1)[..., None]
    depth_loss = ((zz - gt['depth_gt']) ** 2)

    # zz_mean = torch.mean(zz, dim=0, keepdim=True)
    # zz_avg_loss =  ((zz - zz_mean) ** 2)

    if mask is None:
        return {'img_loss':  (img_loss_all + depth_loss + normal_loss).mean()}
    else:
        return {'img_loss': img_loss_all,
                # 'depth_loss': (mask * (depth_loss)).mean(),
                # 'normal_loss':(mask * (depth_loss)).mean(),
                # 'zz_avg_loss': (mask * (zz_avg_loss)).mean()
                }


def render_NL_img_mse_sv_albedo_two_channel(mask, model_output, gt):
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    Laplacian = diff_operators.laplace(model_output['model_out'], model_output['model_in'])
    dx, dy = gradients[:,:,0], gradients[:,:,1]

    xx, yy =model_output['model_in'][:,:,0], model_output['model_in'][:,:,1]
    zz = model_output['model_out'][: , :,0].unsqueeze(2)
    albedo = model_output['model_out'][:, :, 1].unsqueeze(2)
    nx = - dx.unsqueeze(2)
    ny = - dy.unsqueeze(2)
    nz = torch.ones_like(nx)
    normal_set = torch.stack([nx, ny, nz], dim=2).squeeze(3)
    N_norm = torch.norm(normal_set, p=2, dim=2)
    normal_dir = normal_set / N_norm.unsqueeze(2)

    point_set = torch.stack([xx.unsqueeze(2), yy.unsqueeze(2), zz], dim=2).squeeze(3)

    # now we test use the rendering error for all image sequence
    batch_size, numLEDs, _ =  gt['LED_loc'].shape
    img_loss_all = 0
    for i in range(numLEDs):
        LED_loc = gt['LED_loc'][:, i].unsqueeze(1)
        lights = LED_loc - point_set
        L_norm = torch.norm(lights, p=2, dim=2).unsqueeze(2)
        light_dir = lights / L_norm
        light_falloff = torch.pow(L_norm, -2)

        shading =  torch.sum(light_dir * normal_dir, dim=2, keepdims=True)
        attach_shadow = torch.nn.ReLU()
        img = albedo * light_falloff * attach_shadow(shading)
        img_loss = (mask * ((img - gt['img'][:, :, i].unsqueeze(2)) ** 2)).mean()
        img_loss_all =  img_loss_all + img_loss

    normal_loss = 1 - F.cosine_similarity(normal_dir, gt['normal_gt'], dim=-1)[..., None]
    depth_loss = ((zz - gt['depth_gt']) ** 2)

    # zz_mean = torch.mean(zz, dim=0, keepdim=True)
    # zz_avg_loss =  ((zz - zz_mean) ** 2)

    if mask is None:
        return {'img_loss':  (img_loss_all + depth_loss + normal_loss).mean()}
    else:
        return {
                'img_loss': img_loss_all,
                'laplace_loss': (mask * (Laplacian) ** 2).mean()
                # 'depth_loss': (mask * (depth_loss)).mean(),
                # 'normal_loss':(mask * (depth_loss)).mean(),
                # 'zz_avg_loss': (mask * (zz_avg_loss)).mean()
                }


def render_NL_img_sv_albedo_PDE(mask, model_output, gt):
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])

    dx, dy = gradients[:,:,0], gradients[:,:,1]

    xx, yy =model_output['model_in'][:,:,0], model_output['model_in'][:,:,1]
    zz = model_output['model_out']
    # zz = gt['depth_gt']  # for debug

    nx = - dx.unsqueeze(2)
    ny = - dy.unsqueeze(2)
    nz = torch.ones_like(nx)
    normal_set = torch.stack([nx, ny, nz], dim=2).squeeze(3)
    N_norm = torch.norm(normal_set, p=2, dim=2)
    normal_dir = normal_set / N_norm.unsqueeze(2)

    # normal_dir = gt['normal_gt']  # for debug

    point_set = torch.stack([xx.unsqueeze(2), yy.unsqueeze(2), zz], dim=2).squeeze(3)

    # now we test use the rendering error for all image sequence
    batch_size, numLEDs, _ =  gt['LED_loc'].shape
    img_loss_all = 0
    S_p = None
    S_0 = None
    for i in range(numLEDs):
        LED_loc = gt['LED_loc'][:, i].unsqueeze(1)
        lights = LED_loc - point_set
        L_norm = torch.norm(lights, p=2, dim=2).unsqueeze(2)
        light_dir = lights / L_norm
        light_falloff = torch.pow(L_norm, -2)

        shading =  torch.sum(light_dir * normal_dir, dim=2, keepdims=True)
        attach_shadow = torch.nn.ReLU()
        img = light_falloff * attach_shadow(shading)

        if i ==0:
            S_p = torch.clone(img)
            S_0 = torch.clone(img)
        else:
            img_loss = (mask * ((img * gt['img'][:, :, i-1].unsqueeze(2)  - S_p * gt['img'][:, :, i].unsqueeze(2)) ** 2)).mean()
            img_loss_all =  img_loss_all + img_loss
            S_p = torch.clone(img)

    # add ratio of image last and image 0
    img_loss = (mask * (
                (S_0 * gt['img'][:, :, -1].unsqueeze(2) - S_p * gt['img'][:, :, 0].unsqueeze(2)) ** 2)).mean()

    img_loss_all = img_loss_all + img_loss

    normal_loss = 1 - F.cosine_similarity(normal_dir, gt['normal_gt'], dim=-1)[..., None]
    depth_loss = ((zz - gt['depth_gt']) ** 2)

    # zz_mean = torch.mean(zz, dim=0, keepdim=True)
    # zz_avg_loss =  ((zz - zz_mean) ** 2)

    if mask is None:
        return {'img_loss':  (img_loss_all + depth_loss + normal_loss).mean()}
    else:
        return {
                'img_loss': img_loss_all,
                # 'depth_loss': (mask * (depth_loss)).mean(),
                # 'normal_loss':(mask * (depth_loss)).mean(),
                # 'zz_avg_loss': (mask * (zz_avg_loss)).mean()
                }


def render_NL_img_mse_sv_albedo_lstsq_l2(mask, model_output, gt, total_steps, device):
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    dx, dy = gradients[:, :, 0], gradients[:, :, 1]

    xx, yy = model_output['model_in'][:, :, 0], model_output['model_in'][:, :, 1]
    zz = model_output['model_out']
    nx = - dx.unsqueeze(2)
    ny = - dy.unsqueeze(2)
    nz = torch.ones_like(nx)
    normal_set = -torch.stack([nx, ny, nz], dim=2).squeeze(3)
    N_norm = torch.norm(normal_set, p=2, dim=2)
    normal_dir = normal_set / N_norm.unsqueeze(2)

    # normal_dir = gt['normal_gt']
    # zz = gt['depth_gt']

    point_set = torch.stack([xx.unsqueeze(2), yy.unsqueeze(2), zz], dim=2).squeeze(3)

    # now we test use the rendering error for all image sequence
    batch_size, numLEDs, _ = gt['LED_loc'].shape
    batch_size, numPixel, numChannel = zz.shape
    shading_set = torch.zeros([batch_size, numPixel, numChannel, numLEDs], dtype=torch.float64)
    shading_set = shading_set.to(device)
    attach_shadow = torch.nn.ReLU()
    for i in range(numLEDs):
        LED_loc = gt['LED_loc'][:, i].unsqueeze(1)
        lights = LED_loc - point_set
        L_norm = torch.norm(lights, p=2, dim=2).unsqueeze(2)
        light_dir = lights / L_norm
        light_falloff = torch.pow(L_norm, -2)

        shading = torch.sum(light_dir * normal_dir, dim=2, keepdims=True)

        img = light_falloff * shading
        shading_set[:, :, :, i] = img
        # shading_set[:, :, :, i] = torch.where(torch.isnan(img), shading_set[:, :, :, i], img)


    # Calc the albedo from the least square
    if total_steps > 2e3:
        shading_set = attach_shadow(shading_set)

    shading_sum = (shading_set * shading_set).sum(dim = 3)
    shading_sum = torch.where(shading_sum < 1e-8, torch.ones_like(shading_sum) * 1e-8, shading_sum)
    albedo = (gt['img'].unsqueeze(2) * attach_shadow(shading_set)).sum(dim = 3) / shading_sum

    # L2 loss
    residue = gt['img'].unsqueeze(2) - attach_shadow(shading_set) * albedo.unsqueeze(3)
    img_loss_all = (mask * (( (residue) ** 2).mean(dim = 3))).mean()

    # for debug
    # normal_loss = 1 - F.cosine_similarity(normal_dir, gt['normal_gt'], dim=-1)[..., None]
    # depth_loss = ((zz - gt['depth_gt']) ** 2)


    if mask is None:
        return {'img_loss': (img_loss_all + depth_loss + normal_loss).mean()}
    else:
        return {
                'img_loss': img_loss_all,
                # 'depth_loss': (mask * (depth_loss)).mean(),
                # 'normal_loss':(mask * (normal_loss)).mean(),
                }



def debug_depth_normal_loss(mask, model_output, gt, total_steps, device):
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    dx, dy = gradients[:, :, 0], gradients[:, :, 1]

    xx, yy = model_output['model_in'][:, :, 0], model_output['model_in'][:, :, 1]
    zz = model_output['model_out']
    du = dx.unsqueeze(2)
    dv = dy.unsqueeze(2)
    dz = torch.ones_like(du)

    if 'cam_para' in gt:
        focal_len, sensor_height, sensor_width = gt['cam_para'][0]
        sensor_xx, sensor_yy = sensor_width / 2 * xx.unsqueeze(2), sensor_height / 2 * yy.unsqueeze(2)
        dZ_sensor_x, dZ_sensor_y = du * 2 / sensor_width, dv * 2 / sensor_height
        nxp = dZ_sensor_x * focal_len
        nyp = dZ_sensor_y * focal_len
        nzp = - (zz + sensor_xx*dZ_sensor_x + sensor_yy * dZ_sensor_y)
        normal_set = torch.stack([nxp, nyp, nzp], dim=2).squeeze(3)
    else:
        normal_set = torch.stack([du, dv, -dz], dim=2).squeeze(3)
    N_norm = torch.norm(normal_set, p=2, dim=2)
    normal_dir = normal_set / N_norm.unsqueeze(2)

    # for debug
    normal_loss = 1 - F.cosine_similarity(normal_dir, gt['normal_gt'], dim=-1)[..., None]
    depth_loss = ((zz - gt['depth_gt']) ** 2)


    if mask is None:
        return {'img_loss': (depth_loss + normal_loss).mean()}
    else:
        return {
                # 'img_loss': img_loss_all,
                'depth_loss': (mask * (depth_loss)).mean(),
                'normal_loss':(mask * (normal_loss)).mean(),
                }


def render_NL_img_mse_sv_albedo_lstsq_l1(mask, model_output, gt, total_steps, device):
    gradients = diff_operators.gradient(model_output['model_out'], model_output['model_in'])
    dx, dy = gradients[:, :, 0], gradients[:, :, 1]

    xx, yy = model_output['model_in'][:, :, 0], model_output['model_in'][:, :, 1]
    zz = model_output['model_out']
    du = dx.unsqueeze(2)
    dv = dy.unsqueeze(2)
    dz = torch.ones_like(du)

    # for debug
    # zz = gt['depth_gt']

    if 'cam_para' in gt:
        # perspective projection
        focal_len, sensor_height, sensor_width = gt['cam_para'][0]
        sensor_xx, sensor_yy = sensor_width / 2 * xx.unsqueeze(2), sensor_height / 2 * yy.unsqueeze(2)
        dZ_sensor_x, dZ_sensor_y = du * 2 / sensor_width, dv * 2 / sensor_height
        nxp = dZ_sensor_x * focal_len
        nyp = dZ_sensor_y * focal_len
        nzp = - (zz + sensor_xx*dZ_sensor_x + sensor_yy * dZ_sensor_y)
        normal_set = torch.stack([nxp, nyp, nzp], dim=2).squeeze(3)

        point_set = torch.stack([sensor_xx * zz / focal_len,
                                 sensor_yy * zz / focal_len,
                                 zz], dim=2).squeeze(3)

    else:
        # orthographic projection
        normal_set = torch.stack([du, dv, -dz], dim=2).squeeze(3)
        point_set = torch.stack([xx.unsqueeze(2), yy.unsqueeze(2), zz], dim=2).squeeze(3)
    N_norm = torch.norm(normal_set, p=2, dim=2)
    normal_dir = normal_set / N_norm.unsqueeze(2)

    # for debug
    # normal_dir = gt['normal_gt']

    # now we test use the rendering error for all image sequence
    batch_size, numLEDs, _ = gt['LED_loc'].shape
    batch_size, numPixel, numChannel = zz.shape
    shading_set = torch.zeros([batch_size, numPixel, numChannel, numLEDs], dtype=torch.float64)

    shading_set = shading_set.to(device)
    attach_shadow = torch.nn.ReLU()
    for i in range(numLEDs):
        LED_loc = gt['LED_loc'][:, i].unsqueeze(1)
        lights = LED_loc - point_set
        L_norm = torch.norm(lights, p=2, dim=2).unsqueeze(2)
        light_dir = lights / L_norm
        light_falloff = torch.pow(L_norm, -2)

        shading = torch.sum(light_dir * normal_dir, dim=2, keepdims=True)

        img = light_falloff * shading
        shading_set[:, :, :, i] = img
        # shading_set[:, :, :, i] = torch.where(torch.isnan(img), shading_set[:, :, :, i], img)


    # Calc the albedo from the least square
    if total_steps > 2e3:
        shading_set = attach_shadow(shading_set)

    shading_sum = (shading_set * shading_set).sum(dim = 3)
    shading_sum = torch.where(shading_sum < 1e-8, torch.ones_like(shading_sum) * 1e-8, shading_sum)
    albedo = (gt['img'] * attach_shadow(shading_set)).sum(dim = 3) / shading_sum


    # L1 loss
    residue = torch.abs(gt['img'] - attach_shadow(shading_set) * albedo.unsqueeze(3))
    img_loss_all = (mask.unsqueeze(0) * residue.mean(dim = 3)).mean()

    # for debug
    # normal_loss = 1 - F.cosine_similarity(normal_dir, gt['normal_gt'], dim=-1)[..., None]
    # depth_loss = ((zz - gt['depth_gt']) ** 2)


    if mask is None:
        return {'img_loss': (img_loss_all + depth_loss + normal_loss).mean()}
    else:
        return {
                'img_loss': img_loss_all,
                # 'depth_loss': (mask * (depth_loss)).mean(),
                # 'normal_loss':(mask * (normal_loss)).mean(),
                }
