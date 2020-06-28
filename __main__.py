#!/usr/bin/python
# -*- coding: utf8 -*-

"""
Copyright 2018 by Geek3, https://commons.wikimedia.org/wiki/User:Geek3
Licensed under the GNU General Public License 3.0 or later.
"""

from PIL import Image  # install Pillow for PIL
import numpy as np
import scipy.linalg as la
from scipy.special import eval_genlaguerre, lpmv
import sys
from math import *


def hls_to_rgb(h, l, s):
    """
    vectorized function is much faster than point-wise colorsys.hls_to_rgb()
    https://en.wikipedia.org/wiki/HSL_and_HSV#From_HSL
    """
    shape = np.shape(h)
    h = np.ravel(h) % 1.
    l = np.clip(np.ravel(l), 0., 1.)
    s = np.clip(np.ravel(s), 0., 1.)
    h6 = h * 6.
    h6i = h6.astype(np.int)
    C = (1. - np.fabs(2. * l - 1.)) * s
    X = C * (1. - np.fabs(h6 % 2. - 1.))
    m = l - 0.5 * C
    RGB = np.zeros((len(h), 3))
    RGB[:, 0] = np.choose(h6i, [C, X, 0., 0., X, C]) + m
    RGB[:, 1] = np.choose(h6i, [X, C, C, X, 0., 0.]) + m
    RGB[:, 2] = np.choose(h6i, [0., 0., X, C, C, X]) + m
    return RGB.reshape(shape + (3,))


def Rnl(n, l, r):
    """
    radial part of the wavefunction. r may be an array
    """
    rho = np.abs(r) * (2. / n)
    L = eval_genlaguerre(n - l - 1, 2 * l + 1, rho)
    p = factorial(n + l) / factorial(n - l - 1)
    return sqrt((2. / n) ** 3 / (2 * n * p)) * np.exp(-rho / 2.) * rho ** l * L


def psinlm(n, l, m, x, y, z):
    """
    hydrogen atom wavefunction. x, y, z may be arrays of equal length
    fast evaluation, avoiding trigonometric functions alltogether
    """
    assert n >= 1 and 0 <= l < n and -l <= m <= l
    xxyy = x * x + y * y
    rxy = np.sqrt(xxyy)
    r = np.sqrt(xxyy + z * z)
    R = Rnl(n, l, r)

    N = sqrt((2 * l + 1) * factorial(l - m) / (factorial(l + m) * 4 * pi))
    costheta = np.clip(z / r, -1, 1)
    # compute sph_harm() without using trigonometric functions
    Ylm = N * lpmv(m, l, costheta) * ((x + 1j * y) / rxy) ** m
    return R * Ylm


def vnorm(x):
    d = sqrt(np.sum(x * x))
    if d != 0.: return np.array(x) / d
    return np.array(x)


def rtp_to_xyz(rtp):
    st, ct = sin(rtp[1]), cos(rtp[1])
    sp, cp = sin(rtp[2]), cos(rtp[2])
    return rtp[0] * np.array([cp * st, sp * st, ct])


def xyz_to_rtp(xyz):
    r = la.norm(xyz)
    t = atan2(hypot(xyz[0], xyz[1]), xyz[2])
    p = atan2(xyz[1], xyz[0])
    return np.array([r, t, p])


def phong_brightness(vlight, vsurf, vview, ambient=0.35,
                     diffuse=0.4, diffuse_dark=0.2, specular=0.23, specularity=13.):
    """
    phong shading, vsurf and vview may be arrays of vectors
    vlight: vector of light direction
    vsurf: vectors of surface normal vectors (or density gradients)
    vview: vectors of viewing direction
    """
    N = vsurf.shape[1]
    vlight /= la.norm(vlight)
    vsurf /= la.norm(vsurf, axis=0)
    vview /= la.norm(vview, axis=0)
    prod_light_surf = np.dot(vlight, vsurf)
    vreflect = 2. * prod_light_surf * vsurf - np.tile(vlight, (N, 1)).T
    vreflect /= la.norm(vreflect, axis=0)
    amb = ambient * np.ones(N)
    diffuse_frac = np.dot(vlight, vsurf)
    dif = diffuse * diffuse_frac
    dif[diffuse_frac < 0.] = diffuse_dark * diffuse_frac[diffuse_frac < 0.]
    spec = specular * np.maximum(0., np.sum(vview * vreflect, 0)) ** specularity
    # no specular reflection towards the inside of a surface or density gradient
    spec[prod_light_surf < 0.] = 0.
    lightness = np.clip(amb + dif + spec, 0, 1)
    return lightness


def draw_orbital(nlm, w=200, fname=None, density=None, camera_phi=radians(-90),
                 camera_theta=radians(60), light_phi=radians(30), light_theta=0.7,
                 angle_of_view=atan(4.0 / 3.0), view_center=[0, 0, 0], zoom=None):
    """
    creates a pixel graphic of an orbital.
    nlm: either quantum numbers [n, l, m] or a list [[n1, l1, m1, ampl1], ...]
    """

    # shortcut for wavefunction with given parameters
    if type(nlm[0]) == int:
        n, l, m = nlm
        nmax = n

        def psi(x, y, z):
            return psinlm(n, l, m, x, y, z)
    else:
        # mix of different eigenfunctions
        # nlm = [[n1, l1, m1, amplitude1], [n2, l2, m2, amplitude2], ...]
        nmax = max([nlma[0] for nlma in nlm])

        def psi(x, y, z):
            return np.sum([a * psinlm(n, l, m, x, y, z) for n, l, m, a in nlm],
                          axis=0)

    if zoom == None:
        zoom = 1.0 / sqrt(1.0 + 0.6 / nmax ** 2)
    bohr_radii_per_halfwidth = 5.0 * nmax ** 2 / zoom
    h = w  # image size
    unit = w / bohr_radii_per_halfwidth

    # density0 scales overall cloud density
    if density != None:
        density0 = density
    else:
        density0 = 12. * nmax ** 4

    view_center = np.array(view_center)

    # camera location
    camera = view_center + rtp_to_xyz([bohr_radii_per_halfwidth *
                                       sqrt(1.0 + (h / float(w)) ** 2) / tan(radians(angle_of_view) / 2.0),
                                       camera_theta, camera_phi])

    # light source
    dlight = rtp_to_xyz([1.0, light_theta, light_phi])
    vm = view_center - camera
    z0 = np.array([0, 0, 1.0])

    # image plane axes
    image_z = vnorm(vm)
    image_y = vnorm(z0 - np.dot(z0, image_z) * image_z)
    image_x = vnorm(np.cross(image_z, z0))

    # draw
    im = Image.new('RGBA', (w, h))
    for ny in range(h):
        for nx in range(w):
            x, y = (nx - 0.5 * (w - 1)) / unit, (0.5 * (h - 1) - ny) / unit
            p2 = view_center + x * image_x + y * image_y

            # Computation of psi along line of sight has to be vectorized
            # with numpy for reasonable performance.
            Npoints = 501
            tmin = -6 * nmax ** 2
            tmax = 6 * nmax ** 2

            # Use sinh distribution function which samples denser at small
            # radii where the wavefunction has more features.
            tlin = np.linspace(asinh(tmin), asinh(tmax), Npoints)
            tlist = np.sinh(tlin)
            dt = np.cosh(tlin) * (tlin[1] - tlin[0])
            ray_vec = vnorm(p2 - camera)
            points = p2.reshape((3, 1)) + np.outer(ray_vec, tlist)
            # Calls of psi are expensive (around 50 microseconds)
            # So combine the density and gradient calls.
            d_grad = 1e-3
            points_grad = np.tile(points, (1, 4))
            points_grad[0, Npoints:2 * Npoints] += d_grad
            points_grad[1, Npoints * 2:3 * Npoints] += d_grad
            points_grad[2, Npoints * 3:4 * Npoints] += d_grad

            psi_array = psi(points_grad[0], points_grad[1], points_grad[2])
            psi2_array = np.abs(psi_array) ** 2

            density_array = psi2_array[:Npoints]
            # compute density gradient at each point
            grad = (psi2_array[Npoints:4 * Npoints].reshape(3, -1)
                    - density_array) / d_grad
            grad_norm = np.sqrt(np.sum(grad * grad, axis=0))
            grad /= np.maximum(grad_norm, np.finfo(np.float).eps)

            # normalize the gradient
            rr = np.sum(points * points, axis=0)
            gradnorm_rpsi = grad_norm * rr * nmax ** 3

            gradient_scale = 25.
            grad_rel = 1.0 - np.exp(-gradient_scale * gradnorm_rpsi)

            # Color hue is determined by the phase of psi
            phases = np.angle(psi_array[:Npoints])

            # Lightness is determined by density gradient with phong shading
            phongs = phong_brightness(dlight, -grad,
                                      np.tile(camera, (Npoints, 1)).T - points)
            # Put ambient brightness where gradient is low
            ambient = 0.35
            lightness = phongs * grad_rel + ambient * (1. - grad_rel)

            colors = 256. * hls_to_rgb((phases / (2 * pi) - 1. / 3.) % 1.,
                                       lightness, np.ones_like(lightness))

            # Sum up the colors along viewing direction
            dens_integral = np.cumsum(dt * density_array)
            # This is the https://en.wikipedia.org/wiki/Beer-Lambert_law
            opacity = 1. - np.exp(-density0 * dens_integral)
            weights = np.copy(opacity)
            weights[1:] -= opacity[:-1]
            total_opacity = opacity[-1]

            if total_opacity <= 0.:
                total_color = [0., 0., 0.]
                total_opacity = 0.
            else:
                weights /= total_opacity
                total_color = np.dot(weights, colors)

            rgba = np.concatenate((total_color, (256. * total_opacity,)))

            rgba_tuple = tuple(np.clip(rgba, 0, 255).astype('uint8'))
            im.putpixel((nx, ny), rgba_tuple)

        # print status
        outstr = ' row ' + str(ny + 1) + ' of ' + str(h) + ' complete'
        # print('\b{0}{1}'.format(outstr, '\b' * len(outstr)))
        print('\b{0}{1}'.format(outstr, '\b' * len(outstr)), end=' ')
        sys.stdout.flush()

    if fname is None:
        fname = 'hydrogen_n' + str(n) + '_l' + str(l) + '_m' + str(m) + '.png'
    else:
        if len(fname) < 4 or fname[-4] != '.':
            fname += '.png'
    im.save(fname, optimize=1)
    print('image written to', fname)


def main():
    imgsize = 1000
    for n in range(1, 5 + 1):
        for l in range(0, n):
            for m in range(-l, l + 1):
                fname = 'atomic-orbital-cloud_n{}_l{}_m{}.png'.format(n, l, m)
                print(fname)
                nlm = [n, l, m]
                draw_orbital(nlm, w=imgsize, fname=fname)

    for n in range(2, 5 + 1):
        if n >= 2:
            fname = 'atomic-orbital-cloud_n{}_px.png'.format(n)
            print(fname)
            nlm = [[n, 1, 1, -sqrt(0.5)], [n, 1, -1, +sqrt(0.5)]]
            draw_orbital(nlm, w=imgsize, fname=fname)

            fname = 'atomic-orbital-cloud_n{}_py.png'.format(n)
            print(fname)
            nlm = [[n, 1, 1, 1j * sqrt(0.5)], [n, 1, -1, 1j * sqrt(0.5)]]
            draw_orbital(nlm, w=imgsize, fname=fname)

        if n >= 3:
            fname = 'atomic-orbital-cloud_n{}_dxz.png'.format(n)
            print(fname)
            nlm = [[n, 2, 1, -sqrt(0.5)], [n, 2, -1, +sqrt(0.5)]]
            draw_orbital(nlm, w=imgsize, fname=fname)

            fname = 'atomic-orbital-cloud_n{}_dyz.png'.format(n)
            print(fname)
            nlm = [[n, 2, 1, 1j * sqrt(0.5)], [n, 2, -1, 1j * sqrt(0.5)]]
            draw_orbital(nlm, w=imgsize, fname=fname)

            fname = 'atomic-orbital-cloud_n{}_dx^2-y^2.png'.format(n)
            print(fname)
            nlm = [[n, 2, 2, +sqrt(0.5)], [n, 2, -2, +sqrt(0.5)]]
            draw_orbital(nlm, w=imgsize, fname=fname)

            fname = 'atomic-orbital-cloud_n{}_dxy.png'.format(n)
            print(fname)
            nlm = [[n, 2, 2, -1j * sqrt(0.5)], [n, 2, -2, 1j * sqrt(0.5)]]
            draw_orbital(nlm, w=imgsize, fname=fname)

        if n >= 4:
            fname = 'atomic-orbital-cloud_n{}_fxz^2.png'.format(n)
            print(fname)
            nlm = [[n, 3, 1, +sqrt(0.5)], [n, 3, -1, -sqrt(0.5)]]
            draw_orbital(nlm, w=imgsize, fname=fname)

            fname = 'atomic-orbital-cloud_n{}_fyz^2.png'.format(n)
            print(fname)
            nlm = [[n, 3, 1, -1j * sqrt(0.5)], [n, 3, -1, -1j * sqrt(0.5)]]
            draw_orbital(nlm, w=imgsize, fname=fname)

            fname = 'atomic-orbital-cloud_n{}_fz(x^2-y^2).png'.format(n)
            print(fname)
            nlm = [[n, 3, 2, +sqrt(0.5)], [n, 3, -2, +sqrt(0.5)]]
            draw_orbital(nlm, w=imgsize, fname=fname)

            fname = 'atomic-orbital-cloud_n{}_fxyz.png'.format(n)
            print(fname)
            nlm = [[n, 3, 2, -1j * sqrt(0.5)], [n, 3, -2, 1j * sqrt(0.5)]]
            draw_orbital(nlm, w=imgsize, fname=fname)

            fname = 'atomic-orbital-cloud_n{}_fx(x^2-3y^2).png'.format(n)
            print(fname)
            nlm = [[n, 3, 3, -sqrt(0.5)], [n, 3, -3, +sqrt(0.5)]]
            draw_orbital(nlm, w=imgsize, fname=fname)

            fname = 'atomic-orbital-cloud_n{}_fy(y^2-3x^2).png'.format(n)
            print(fname)
            nlm = [[n, 3, 3, 1j * sqrt(0.5)], [n, 3, -3, 1j * sqrt(0.5)]]
            draw_orbital(nlm, w=imgsize, fname=fname)


if __name__ == '__main__':
    main()
