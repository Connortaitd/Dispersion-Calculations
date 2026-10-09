"""
SPECTRAL PHASE / DISPERSION CALCULATOR
======================================

Two pulses:

1. NOPA:
       680 - 950 nm
       5 m air
       1 mm physical-thickness fused silica beamsplitter
       at 45 degrees external incidence

2. Adiabatic OPA:
       approximately 1100 - 2200 nm
       5 m air
       2 mm fused silica (UVFS) filter at normal incidence
       1 mm physical-thickness CaF2 beamsplitter
       at 45 degrees external incidence


The program calculates:

    n(lambda)
    transmitted angle
    optical path length through each beamsplitter
    accumulated spectral phase
    dispersion-only spectral phase
    group delay
    GDD
    TOD
    FOD
    individual contributions from:

        - air
        - fused silica
        - CaF2

The dispersion-only phase has the constant and linear phase
terms removed:

    phi_disp(omega)
        = phi(omega)
          - phi(omega0)
          - (omega-omega0)*dphi/domega|omega0

This is the phase relevant to pulse chirp.

All wavelengths are VACUUM wavelengths.


"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# CONSTANTS
# ============================================================

c = 299792458.0                 # speed of light [m/s]
pi = np.pi


# ============================================================
# OPTICAL GEOMETRY
# ============================================================

AIR_LENGTH_M = 5


# ------------------------------------------------------------
# NOPA beamsplitter
# ------------------------------------------------------------

# Physical thickness measured perpendicular to the surfaces.
NOPA_BS_THICKNESS_M = 1e-3      # 1 mm fused silica

# External angle of incidence in air.
NOPA_BS_ANGLE_DEG = 45.0
NOPA_BS_ANGLE_RAD = np.deg2rad(
    NOPA_BS_ANGLE_DEG
)


# ------------------------------------------------------------
# Adiabatic OPA optics
# ------------------------------------------------------------

# Existing normal-incidence fused silica filter.
FS_FILTER_THICKNESS_M = 2e-3    # 2 mm


# CaF2 beamsplitter.
CAF2_BS_THICKNESS_M = 1e-3      # 1 mm physical thickness

CAF2_BS_ANGLE_DEG = 45.0
CAF2_BS_ANGLE_RAD = np.deg2rad(
    CAF2_BS_ANGLE_DEG
)


# ============================================================
# SELLMEIER EQUATION
# ============================================================

def sellmeier(lam_um, B, C):
    """
    Sellmeier equation:

        n^2 = 1 + sum[
            B_i * lambda^2 / (lambda^2 - C_i)
        ]

    lambda is in micrometers.
    """

    lam_um = np.asarray(
        lam_um,
        dtype=float
    )

    n2 = np.ones_like(lam_um)

    for Bi, Ci in zip(B, C):

        n2 += (
            Bi
            * lam_um**2
            / (lam_um**2 - Ci)
        )

    return np.sqrt(n2)


# ============================================================
# FUSED SILICA
# ============================================================

# Malitson fused silica
#
# Valid approximately from 0.21 to 6.7 um.

FS_B = np.array([
    0.6961663,
    0.4079426,
    0.8974794
])

FS_C = np.array([
    0.0684043**2,
    0.1162414**2,
    9.896161**2
])


def n_fused_silica(lam_um):

    return sellmeier(
        lam_um,
        FS_B,
        FS_C
    )


# ============================================================
# CALCIUM FLUORIDE
# ============================================================

# Malitson CaF2
#
# Valid approximately from 0.23 to 9.7 um.

CAF2_B = np.array([
    0.5675888,
    0.4710914,
    3.8484723
])

CAF2_C = np.array([
    0.050263605**2,
    0.1003909**2,
    34.649040**2
])


def n_caf2(lam_um):

    return sellmeier(
        lam_um,
        CAF2_B,
        CAF2_C
    )


# ============================================================
# AIR
# ============================================================

def n_air(lam_um, temperature_C=21.5, relative_humidity=30,
          pressure_Pa=101325.0):
    """
    Calculate the refractive index of moist air.

    Parameters
    ----------
    lam_um : array_like
        Vacuum wavelength in micrometers.
    temperature_C : float or array_like, optional
        Air temperature in degrees Celsius.
    relative_humidity : float or array_like, optional
        Relative humidity in percent, from 0 to 100.
    pressure_Pa : float or array_like, optional
        Total air pressure in Pa.

    Returns
    -------
    n : ndarray or scalar
        Refractive index of moist air.

    Notes
    -----
    Uses the Edlén 1966 dispersion equation for standard dry air,
    with density scaling for temperature/pressure and a correction
    for water vapor.

    This is suitable for typical atmospheric conditions. For
    high-precision metrology, use the full Ciddor equation.
    """

    lam_um = np.asarray(lam_um, dtype=float)
    T_C = np.asarray(temperature_C, dtype=float)
    RH = np.asarray(relative_humidity, dtype=float)
    P = np.asarray(pressure_Pa, dtype=float)

    if np.any(RH < 0) or np.any(RH > 100):
        raise ValueError("relative_humidity must be between 0 and 100 percent.")

    # Wavenumber in um^-1
    sigma = 1.0 / lam_um

    # Edlén equation for standard dry air.
    n_minus_1_standard = 1e-8 * (
        8342.13
        + 2406030.0 / (130.0 - sigma**2)
        + 15997.0 / (38.9 - sigma**2)
    )

    # Temperature in kelvin
    T_K = T_C + 273.15

    # Saturation vapor pressure over water.
    # Magnus formula, pressure in Pa.
    e_s = 611.21 * np.exp(
        (17.502 * T_C) / (240.97 + T_C)
    )

    # Actual water-vapor partial pressure.
    e = (RH / 100.0) * e_s

    # Dry-air partial pressure.
    p_d = P - e

    # Standard reference conditions for the Edlén dry-air value.
    T_ref = 288.15       # K (15 °C)
    P_ref = 101325.0     # Pa

    # Scale dry-air refractivity approximately with density.
    n_dry_minus_1 = (
        n_minus_1_standard
        * (p_d / P_ref)
        * (T_ref / T_K)
    )

    # Water-vapor refractivity.
    #
    # Approximate dispersion of water vapor in the visible/near-IR.
    n_water_minus_1 = (
        1e-8
        * (
            295.235
            + 2.6422 * sigma**2
            - 0.032380 * sigma**4
            + 0.004028 * sigma**6
        )
    )

    # Scale water-vapor refractivity by water-vapor density.
    n_water_minus_1 *= (e / P_ref) * (T_ref / T_K)

    return 1.0 + n_dry_minus_1 + n_water_minus_1


# ============================================================
# BASIC OPTICAL FUNCTIONS
# ============================================================

def wavelength_to_frequency(lam_m):

    return c / lam_m


def wavelength_to_omega(lam_m):

    return (
        2.0
        * pi
        * c
        / lam_m
    )


# ============================================================
# NORMAL-INCIDENCE PHASE
# ============================================================

def phase_normal_incidence(
        lam_um,
        n_function,
        thickness_m):
    """
    Phase accumulated through a material at normal incidence.

        phi = k0 * n * L

    where

        k0 = 2*pi/lambda

    thickness_m is the physical thickness.
    """

    lam_m = lam_um * 1e-6

    n = n_function(lam_um)

    return (
        2.0
        * pi
        * n
        * thickness_m
        / lam_m
    )


# ============================================================
# AIR PHASE
# ============================================================

def phase_air(
        lam_um,
        length_m):
    """
    Phase accumulated through air.
    """

    return phase_normal_incidence(
        lam_um,
        n_air,
        length_m
    )


# ============================================================
# SNELL'S LAW
# ============================================================

def transmitted_angle(
        lam_um,
        n_function,
        theta_i):
    """
    Snell's law for propagation from air into the material:

        n_air * sin(theta_i)
            = n_material * sin(theta_t)

    Here the incident medium is approximated as vacuum/air
    with n = 1.

        sin(theta_t) = sin(theta_i) / n

    theta_i:
        external angle of incidence

    theta_t:
        internal angle relative to the surface normal
    """

    n = n_function(lam_um)

    sin_theta_t = (
        np.sin(theta_i)
        / n
    )

    return np.arcsin(
        sin_theta_t
    )


# ============================================================
# OBLIQUE PLATE PATH LENGTH
# ============================================================

def oblique_path_length(
        lam_um,
        n_function,
        thickness_m,
        theta_i):
    """
    Calculate the actual distance traveled by the transmitted
    beam inside a plane-parallel optic.

    thickness_m is the physical surface-normal thickness.

        L_path = d / cos(theta_t)

    where theta_t is the refracted angle inside the optic.

    Returned in meters.
    """

    theta_t = transmitted_angle(
        lam_um,
        n_function,
        theta_i
    )

    return (
        thickness_m
        / np.cos(theta_t)
    )


# ============================================================
# OBLIQUE PLATE PHASE
# ============================================================

def phase_oblique_plate(
        lam_um,
        n_function,
        thickness_m,
        theta_i):
    """
    Spectral phase accumulated by a transmitted beam through
    an oblique plane-parallel optic.

    thickness_m:
        physical thickness measured perpendicular to the
        optic surfaces.

    theta_i:
        external angle of incidence.

    First calculate the internal angle using Snell's law:

        sin(theta_t) = sin(theta_i) / n

    Then the actual beam path through the material is:

        L_path = thickness_m / cos(theta_t)

    The accumulated phase along the beam path is:

        phi = k0 * n * L_path

            = (2*pi/lambda)
              * n
              * thickness_m / cos(theta_t)
    """

    lam_m = lam_um * 1e-6

    n = n_function(lam_um)

    path_length = oblique_path_length(
        lam_um,
        n_function,
        thickness_m,
        theta_i
    )

    return (
        2.0
        * pi
        / lam_m
        * n
        * path_length
    )


# ============================================================
# DISPERSION-ONLY PHASE
# ============================================================

def remove_constant_and_linear_phase(
        omega,
        phase):
    """
    Remove:

        1. Constant phase
        2. Linear spectral phase

    The remaining phase is the dispersion-only phase.

    This is equivalent to choosing the reference frequency
    at the center of the spectrum and setting:

        phi(omega0) = 0
        dphi/domega(omega0) = 0
    """

    omega = np.asarray(
        omega
    )

    phase = np.asarray(
        phase
    )

    center_index = len(omega) // 2

    omega0 = omega[
        center_index
    ]

    phi0 = phase[
        center_index
    ]

    dphi_domega = np.gradient(
        phase,
        omega
    )

    slope = dphi_domega[
        center_index
    ]

    phase_disp = (
        phase
        - phi0
        - slope * (
            omega - omega0
        )
    )

    return phase_disp


# ============================================================
# GROUP DELAY
# ============================================================

def group_delay(
        omega,
        phase):
    """
    Group delay:

        tau_g = d(phi)/d(omega)

    Returned in femtoseconds.
    """

    dphi_domega = np.gradient(
        phase,
        omega
    )

    return (
        dphi_domega
        * 1e15
    )


# ============================================================
# GDD
# ============================================================

def calculate_gdd(
        omega,
        phase):
    """
    GDD:

        GDD = d^2(phi)/d(omega)^2

    Returned in fs^2.
    """

    first_derivative = np.gradient(
        phase,
        omega
    )

    second_derivative = np.gradient(
        first_derivative,
        omega
    )

    return (
        second_derivative
        * 1e30
    )


# ============================================================
# TOD
# ============================================================

def calculate_tod(
        omega,
        phase):
    """
    TOD:

        TOD = d^3(phi)/d(omega)^3

    Returned in fs^3.
    """

    first_derivative = np.gradient(
        phase,
        omega
    )

    second_derivative = np.gradient(
        first_derivative,
        omega
    )

    third_derivative = np.gradient(
        second_derivative,
        omega
    )

    return (
        third_derivative
        * 1e45
    )


# ============================================================
# FOD
# ============================================================

def calculate_fod(
        omega,
        phase):
    """
    FOD:

        FOD = d^4(phi)/d(omega)^4

    Returned in fs^4.
    """

    first_derivative = np.gradient(
        phase,
        omega
    )

    second_derivative = np.gradient(
        first_derivative,
        omega
    )

    third_derivative = np.gradient(
        second_derivative,
        omega
    )

    fourth_derivative = np.gradient(
        third_derivative,
        omega
    )

    return (
        fourth_derivative
        * 1e60
    )


# ============================================================
# CALCULATE ONE PULSE
# ============================================================

def calculate_pulse(
        wavelength_nm,
        optic_material=None,
        optic_thickness_m=0.0,
        optic_angle_rad=0.0,
        include_filter=False):
    """
    Calculate all spectral phase contributions.

    Parameters
    ----------
    wavelength_nm : array
        Vacuum wavelength in nm.

    optic_material : function or None
        Refractive-index function for the oblique optic.

        Examples:
            n_fused_silica
            n_caf2

    optic_thickness_m : float
        Physical surface-normal thickness of the oblique
        optic.

    optic_angle_rad : float
        External angle of incidence.

    include_filter : bool
        If True, add the 2 mm normal-incidence fused silica
        filter used by the Adiabatic OPA.
    """

    wavelength_nm = np.asarray(
        wavelength_nm,
        dtype=float
    )

    wavelength_um = (
        wavelength_nm / 1000.0
    )

    wavelength_m = (
        wavelength_um * 1e-6
    )

    omega = wavelength_to_omega(
        wavelength_m
    )

    # --------------------------------------------------------
    # AIR
    # --------------------------------------------------------

    phi_air = phase_air(
        wavelength_um,
        AIR_LENGTH_M
    )

    # --------------------------------------------------------
    # OBLIQUE BEAMSPLITTER
    # --------------------------------------------------------

    if optic_material is not None:

        phi_oblique = phase_oblique_plate(
            wavelength_um,
            optic_material,
            optic_thickness_m,
            optic_angle_rad
        )

        transmitted_angle_rad = transmitted_angle(
            wavelength_um,
            optic_material,
            optic_angle_rad
        )

        path_length_m = oblique_path_length(
            wavelength_um,
            optic_material,
            optic_thickness_m,
            optic_angle_rad
        )

    else:

        phi_oblique = np.zeros_like(
            wavelength_um
        )

        transmitted_angle_rad = np.zeros_like(
            wavelength_um
        )

        path_length_m = np.zeros_like(
            wavelength_um
        )

    # --------------------------------------------------------
    # 2 mm FUSED SILICA FILTER
    # --------------------------------------------------------

    if include_filter:

        phi_fs = phase_normal_incidence(
            wavelength_um,
            n_fused_silica,
            FS_FILTER_THICKNESS_M
        )

    else:

        phi_fs = np.zeros_like(
            wavelength_um
        )

    # --------------------------------------------------------
    # TOTAL PHASE
    # --------------------------------------------------------

    phi_total = (
        phi_air
        + phi_oblique
        + phi_fs
    )

    # --------------------------------------------------------
    # DISPERSION-ONLY PHASES
    # --------------------------------------------------------

    phi_air_disp = (
        remove_constant_and_linear_phase(
            omega,
            phi_air
        )
    )

    phi_oblique_disp = (
        remove_constant_and_linear_phase(
            omega,
            phi_oblique
        )
    )

    phi_fs_disp = (
        remove_constant_and_linear_phase(
            omega,
            phi_fs
        )
    )

    phi_total_disp = (
        remove_constant_and_linear_phase(
            omega,
            phi_total
        )
    )

    # --------------------------------------------------------
    # GROUP DELAY
    # --------------------------------------------------------

    group_delay_total = group_delay(
        omega,
        phi_total
    )

    # --------------------------------------------------------
    # GDD
    # --------------------------------------------------------

    gdd_total = calculate_gdd(
        omega,
        phi_total
    )

    # --------------------------------------------------------
    # TOD
    # --------------------------------------------------------

    tod_total = calculate_tod(
        omega,
        phi_total
    )

    # --------------------------------------------------------
    # FOD
    # --------------------------------------------------------

    fod_total = calculate_fod(
        omega,
        phi_total
    )

    return {

        "wavelength_nm":
            wavelength_nm,

        "wavelength_um":
            wavelength_um,

        "omega":
            omega,

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        "transmitted_angle_rad":
            transmitted_angle_rad,

        "transmitted_angle_deg":
            np.rad2deg(
                transmitted_angle_rad
            ),

        "oblique_path_length_m":
            path_length_m,

        # ----------------------------------------------------
        # Raw phases
        # ----------------------------------------------------

        "phase_air":
            phi_air,

        "phase_oblique":
            phi_oblique,

        "phase_fs":
            phi_fs,

        "phase_total":
            phi_total,

        # ----------------------------------------------------
        # Dispersion-only phases
        # ----------------------------------------------------

        "phase_air_disp":
            phi_air_disp,

        "phase_oblique_disp":
            phi_oblique_disp,

        "phase_fs_disp":
            phi_fs_disp,

        "phase_total_disp":
            phi_total_disp,

        # ----------------------------------------------------
        # Temporal quantities
        # ----------------------------------------------------

        "group_delay_fs":
            group_delay_total,

        "gdd_fs2":
            gdd_total,

        "tod_fs3":
            tod_total,

        "fod_fs4":
            fod_total
    }


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
        result,
        wavelengths_to_print,
        optic_name=None):
    """
    Print dispersion information at selected wavelengths.
    """

    wavelength = result[
        "wavelength_nm"
    ]

    print("\n============================================")
    print("DISPERSION SUMMARY")
    print("============================================")

    if optic_name is not None:

        print(
            f"Oblique optic: {optic_name}"
        )

    for target in wavelengths_to_print:

        index = np.argmin(
            np.abs(
                wavelength - target
            )
        )

        print(
            f"\nWavelength = "
            f"{wavelength[index]:.3f} nm"
        )

        print(
            f"Internal angle = "
            f"{result['transmitted_angle_deg'][index]:.6f} deg"
        )

        print(
            f"Beam path through oblique optic = "
            f"{result['oblique_path_length_m'][index] * 1e3:.6f} mm"
        )

        print(
            f"Total group delay = "
            f"{result['group_delay_fs'][index]:.6f} fs"
        )

        print(
            f"Total GDD = "
            f"{result['gdd_fs2'][index]:.6f} fs^2"
        )

        print(
            f"Total TOD = "
            f"{result['tod_fs3'][index]:.6f} fs^3"
        )

        print(
            f"Total FOD = "
            f"{result['fod_fs4'][index]:.6f} fs^4"
        )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # NOPA
    # ========================================================

    nopa_wavelength = np.linspace(
        680.0,
        950.0,
        400
    )

    nopa = calculate_pulse(

        nopa_wavelength,

        # 1 mm PHYSICAL fused silica
        # beamsplitter at 45 degrees
        optic_material=n_fused_silica,

        optic_thickness_m=
            NOPA_BS_THICKNESS_M,

        optic_angle_rad=
            NOPA_BS_ANGLE_RAD,

        # No 2 mm filter
        include_filter=False
    )

    print("\n\n")
    print("################################################")
    print("# NOPA: 680-950 nm")
    print("# 5 m air")
    print("# 1 mm fused silica beamsplitter @ 45 deg")
    print("################################################")

    print_summary(
        nopa,
        [745],
        optic_name="1 mm fused silica"
    )


    # ========================================================
    # ADIABATIC OPA
    # ========================================================

    pump_wavelength = 515.0

    # Difference-frequency relation used in the original code:
    #
    #   1/lambda_OPA
    #       = 1/lambda_pump - 1/lambda_NOPA

    opa_wavelength = np.array(
        1.0 / (
            1.0 / pump_wavelength
            - 1.0 / nopa_wavelength
        )
    )

    opa = calculate_pulse(

        opa_wavelength,

        # 1 mm PHYSICAL CaF2
        # beamsplitter at 45 degrees
        optic_material=n_caf2,

        optic_thickness_m=
            CAF2_BS_THICKNESS_M,

        optic_angle_rad=
            CAF2_BS_ANGLE_RAD,

        # Existing 2 mm normal-incidence
        # fused silica filter
        include_filter=True
    )

    print("\n\n")
    print("################################################")
    print("# ADIABATIC OPA")
    print("# 5 m air")
    print("# 2 mm fused silica filter")
    print("# 1 mm CaF2 beamsplitter @ 45 deg")
    print("################################################")

    opa_center_wavelength = (
        1.0 / (
            1.0 / pump_wavelength
            - 1.0 / 745.0
        )
    )

    print_summary(
        opa,
        [opa_center_wavelength],
        optic_name="1 mm CaF2"
    )


    #%%
    # ========================================================
    # PLOT 1A:
    # TOTAL DISPERSION-ONLY PHASE FOR NOPA
    # ========================================================

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    ax1.plot(
        nopa["wavelength_nm"],
        nopa["phase_total_disp"],
        linewidth=2,
        label="NOPA"
    )

    ax1.set_xlabel(
        "Wavelength (nm)"
    )

    ax1.set_ylabel(
        "Dispersion-only spectral phase (rad)"
    )

    ax1.set_title(
        "NOPA Phase vs. Wavelength"
    )

    ax1.grid(
        True,
        alpha=0.3
    )

    ax1.legend()

    ax2.plot(
        nopa["omega"] * 1e-15,
        nopa["phase_total_disp"],
        linewidth=2,
        color="orange",
        label="NOPA"
    )

    ax2.set_xlabel(
        r"Omega (rad/sec) ($\times 10^{15}$)"
    )

    ax2.set_ylabel(
        "Dispersion-only spectral phase (rad)"
    )

    ax2.set_title(
        "NOPA Phase vs. Omega"
    )

    ax2.grid(
        True,
        alpha=0.3
    )

    ax2.legend()

    plt.tight_layout()
    plt.show()


    #%%
    # ========================================================
    # PLOT 1B:
    # TOTAL DISPERSION-ONLY PHASE FOR AOPA
    # ========================================================

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    ax1.plot(
        opa["wavelength_nm"],
        opa["phase_total_disp"],
        linewidth=2,
        label="Adiabatic OPA"
    )

    ax1.set_xlabel(
        "Wavelength (nm)"
    )

    ax1.set_ylabel(
        "Dispersion-only spectral phase (rad)"
    )

    ax1.set_title(
        "Adiabatic OPA Phase vs. Wavelength"
    )

    ax1.grid(
        True,
        alpha=0.3
    )

    ax1.legend()

    ax2.plot(
        opa["omega"] * 1e-15,
        opa["phase_total_disp"],
        linewidth=2,
        color="orange",
        label="Adiabatic OPA"
    )

    ax2.set_xlabel(
        r"Omega (rad/sec) ($\times 10^{15}$)"
    )

    ax2.set_ylabel(
        "Dispersion-only spectral phase (rad)"
    )

    ax2.set_title(
        "Adiabatic OPA Phase vs. Omega"
    )

    ax2.grid(
        True,
        alpha=0.3
    )

    ax2.legend()

    plt.tight_layout()
    plt.show()


    #%%
    # ========================================================
    # PLOT 2:
    # NOPA INDIVIDUAL CONTRIBUTIONS
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        nopa["wavelength_nm"],
        nopa["phase_air_disp"],
        label="5 m air"
    )

    plt.plot(
        nopa["wavelength_nm"],
        nopa["phase_oblique_disp"],
        label="1 mm fused silica @ 45°"
    )

    plt.plot(
        nopa["wavelength_nm"],
        nopa["phase_total_disp"],
        linewidth=2,
        label="Total"
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Dispersion-only phase (rad)"
    )

    plt.title(
        "NOPA Spectral Phase Contributions"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


    # ========================================================
    # PLOT 3:
    # AOPA INDIVIDUAL CONTRIBUTIONS
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        opa["wavelength_nm"],
        opa["phase_air_disp"],
        label="5 m air"
    )

    plt.plot(
        opa["wavelength_nm"],
        opa["phase_fs_disp"],
        label="2 mm fused silica"
    )

    plt.plot(
        opa["wavelength_nm"],
        opa["phase_oblique_disp"],
        label="1 mm CaF2 @ 45°"
    )

    plt.plot(
        opa["wavelength_nm"],
        opa["phase_total_disp"],
        linewidth=2,
        label="Total"
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Dispersion-only phase (rad)"
    )

    plt.title(
        "Adiabatic OPA Spectral Phase Contributions"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()
    plt.show()


    #%%
    # ========================================================
    # PLOT 4:
    # DAZZLER PHASE
    # ========================================================

    # The AOPA phase is evaluated on its own wavelength grid.
    #
    # The OPA wavelength grid is generated from the NOPA
    # wavelength grid using the 515 nm pump, so the arrays
    # correspond point-by-point.

    opa_phase_transposed = (
        opa["phase_total_disp"]
    )

    phi_dazzler = (
        nopa["phase_total_disp"]
        - opa_phase_transposed
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10, 6)
    )

    # --------------------------------------------------------
    # Wavelength
    # --------------------------------------------------------

    ax1.plot(
        nopa["wavelength_nm"],
        nopa["phase_total_disp"],
        linewidth=2,
        label="NOPA phase"
    )

    ax1.plot(
        nopa["wavelength_nm"],
        opa_phase_transposed,
        linewidth=2,
        label="Adiabatic OPA phase (transposed)"
    )

    ax1.plot(
        nopa["wavelength_nm"],
        phi_dazzler,
        linewidth=2,
        label="Dazzler phase"
    )

    ax1.plot(
        nopa["wavelength_nm"],
        phi_dazzler
        - nopa["phase_total_disp"]
        + opa_phase_transposed,
        linewidth=2,
        label="Total phase"
    )

    ax1.set_xlabel(
        "Wavelength (nm)"
    )

    ax1.set_ylabel(
        "Dispersion-only spectral phase (rad)"
    )

    ax1.set_title(
        "Phase Compensation"
    )

    ax1.grid(
        True,
        alpha=0.3
    )

    ax1.legend()


    # --------------------------------------------------------
    # Omega
    # --------------------------------------------------------

    ax2.plot(
        nopa["omega"],
        -nopa["phase_total_disp"],
        linewidth=2,
        label="NOPA phase"
    )

    ax2.plot(
        nopa["omega"],
        opa_phase_transposed,
        linewidth=2,
        label="Adiabatic OPA phase (transposed)"
    )

    ax2.plot(
        nopa["omega"],
        phi_dazzler,
        linewidth=2,
        label="Dazzler phase"
    )

    ax2.plot(
        nopa["omega"],
        phi_dazzler
        - nopa["phase_total_disp"]
        + opa_phase_transposed,
        linewidth=2,
        label="Total phase"
    )

    ax2.set_xlabel(
        r"Omega (rad/sec) ($\times 10^{15}$)"
    )

    ax2.set_ylabel(
        "Dispersion-only spectral phase (rad)"
    )

    ax2.set_title(
        "Phase Compensation vs. Omega"
    )

    ax2.grid(
        True,
        alpha=0.3
    )

    ax2.legend()

    plt.tight_layout()
    plt.show()


    #%%
    # ========================================================
    # POLYNOMIAL FIT TO DAZZLER PHASE
    # ========================================================

    wavFit = 745e-9

    angFreqFit = (
        2
        * np.pi
        * c
        / wavFit
    )

    omegaFit = (
        nopa["omega"]
        - angFreqFit
    )

    phaseFit = np.polyfit(
        omegaFit,
        phi_dazzler,
        4
    )

    fitFunction = np.poly1d(
        phaseFit
    )

    print("\n============================================")
    print("DAZZLER POLYNOMIAL FIT")
    print("============================================")

    print(
        "GDD:",
        2
        * phaseFit[-3]
        * 1e30,
        "fs**2"
    )

    print(
        "TOD:",
        6
        * phaseFit[-4]
        * 1e45,
        "fs**3"
    )

    print(
        "FOD:",
        24
        * phaseFit[-5]
        * 1e60,
        "fs**4"
    )


    print(
        "Polynomial coefficients "
        "(highest order first):"
    )

    print(
        phaseFit
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        nopa["omega"],
        phi_dazzler,
        label="Dazzler phase"
    )

    plt.plot(
        nopa["omega"],
        fitFunction(
            nopa["omega"]
            - angFreqFit
        ),
        label="5th-order polynomial fit"
    )

    plt.xlabel(
        r"Omega (rad/sec)"
    )

    plt.ylabel(
        "Phase (rad)"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()
    plt.show()


    #%%
    # ========================================================
    # PLOT 5:
    # GROUP DELAY
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        nopa["wavelength_nm"],
        nopa["group_delay_fs"],
        label="NOPA"
    )

    plt.plot(
        opa["wavelength_nm"],
        opa["group_delay_fs"],
        label="Adiabatic OPA"
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Group delay (fs)"
    )

    plt.title(
        "Group Delay vs Wavelength"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


    # ========================================================
    # DERIVATIVE PLOT RANGE
    # ========================================================

    # Avoid the edges where numerical high-order derivatives
    # become less reliable.

    low_index = int(
        0.05
        * len(
            nopa["wavelength_nm"]
        )
    )

    high_index = int(
        0.95
        * len(
            nopa["wavelength_nm"]
        )
    )


    # ========================================================
    # PLOT 6:
    # GDD
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        nopa["wavelength_nm"][
            low_index:high_index
        ],
        nopa["gdd_fs2"][
            low_index:high_index
        ],
        label="NOPA"
    )

    plt.plot(
        opa["wavelength_nm"][
            low_index:high_index
        ],
        opa["gdd_fs2"][
            low_index:high_index
        ],
        label="Adiabatic OPA"
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        r"GDD (fs$^2$)"
    )

    plt.title(
        "Group-Delay Dispersion"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


    # ========================================================
    # PLOT 7:
    # TOD
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        nopa["wavelength_nm"][
            low_index:high_index
        ],
        nopa["tod_fs3"][
            low_index:high_index
        ],
        label="NOPA"
    )

    plt.plot(
        opa["wavelength_nm"][
            low_index:high_index
        ],
        opa["tod_fs3"][
            low_index:high_index
        ],
        label="Adiabatic OPA"
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        r"TOD (fs$^3$)"
    )

    plt.title(
        "Third-Order Dispersion"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


    # ========================================================
    # PLOT 8:
    # FOD
    # ========================================================

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        nopa["wavelength_nm"][
            low_index:high_index
        ],
        nopa["fod_fs4"][
            low_index:high_index
        ],
        label="NOPA"
    )

    plt.plot(
        opa["wavelength_nm"][
            low_index:high_index
        ],
        opa["fod_fs4"][
            low_index:high_index
        ],
        label="Adiabatic OPA"
    )

    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        r"FOD (fs$^4$)"
    )

    plt.title(
        "Fourth-Order Dispersion"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()
