import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import Planck, speed_of_light, Avogadro


h_bar = Planck / (2*np.pi)
Na = Avogadro
c = speed_of_light * 1e2  # [cm/sec]

m_H = 1.008 / (Na*1e3)  # [kg]
m_O = 15.999 / (Na*1e3)  # [kg]
mu = (m_H * m_O) / (m_H + m_O)  # [kg]


# Kinetic energy
def T(i, j, delta):
    kinetic_coeff = (h_bar**2 / (2*mu*(delta ** 2))) * ((-1)**(i-j))
    if i == j:
        return kinetic_coeff * ((np.pi ** 2)/3 - 1/(2*(i**2)))
    return kinetic_coeff * (2/((i-j)**2) - 2/((i+j)**2))


# Morse potential
def U(r_i):
    return D_e * ((1 - np.exp(-a * (r_i-r_e)))**2 - 1)


def T_matrix(r_vec):
    delta = r_vec[1]-r_vec[0]
    matrix = np.zeros((len(r_vec), len(r_vec)))
    for i in range(0, len(r_vec)):
        for j in range(0, len(r_vec)):
            matrix[i][j] = T(i+1, j+1, delta)
    return matrix


def U_matrix(r_vec):
    matrix = np.zeros((len(r_vec), len(r_vec)))
    for i in range(0, len(r_vec)):
        matrix[i][i] = U(r_vec[i])
    return matrix


def get_eigenstates(matrix):
    energies_tmp, states_tmp = np.linalg.eig(matrix)
    sorted_indeces = np.argsort(energies_tmp)
    sorted_energies = np.array([energies[index] for index in sorted_indeces])
    sorted_states = np.array([states_tmp[:, index] for index in sorted_indeces])
    return sorted_energies, sorted_states


def plot_energies(r, energies):
    plt.figure()

    # plot Morse potential
    plt.plot(r, U(r), color='black', label='Morse Potential')
    plt.legend()

    # plot eigenvalues
    for energy in energies:
        plt.plot(r, np.ones(len(r)) * energy)

    plt.grid()
    plt.xlabel(r'$r [m]$')
    plt.ylabel(r'$Energy [J/molecule]$')
    plt.title("Spectrum")


def plot_wavefuncs(r, energies, vectors, title):
    energy_scale = (energies[1] - energies[0]) / 2

    plt.figure()

    plt.plot(r, U(r), color='black')
    plt.legend(["Morse Potential"])

    for i in range(0, len(vectors)):
        plt.plot(r, ((energy_scale * vectors[i] + energies[i])))
    plt.grid()
    plt.xlabel(r'$r [m]$')
    plt.ylabel(r'$\Psi_v(r) $')
    plt.title('Wavefunctions ' + title)


def U_cent(r, J):
    return (h_bar**2)*J*(J+1)/(2*mu*(r**2))


def U_cent_matrix(r, J):
	matrix = np.zeros((len(r), len(r)))
	for i in range(len(r)):
		matrix[i][i] = U_cent(r[i], J)
	return matrix


def franck_condon(phi1, phi2):
	return np.abs(np.trapz(np.conjugate(phi1)*phi2))**2


def partA_1(r, energies, states, m):
    m_energies = np.array(energies[:m])
    m_states = np.array(states[:][:m])
    print(f'Lowest {m} energies:')
    print(m_energies)

    # plot eigenvalues
    plot_energies(r, m_energies)
    plt.xlim(0.5e-10, 0.5e-9)
    plt.ylim(-0.9e-18, 0.2e-18)

    # plot wavefunctions
    plot_wavefuncs(r, m_energies, m_states, "J=0")
    plt.xlim(0.5e-10, 0.5e-9)
    plt.ylim(-0.9e-18, 0.2e-18)


def partA_2(r, energies, states):
    unbound_index = -1
    i = 0
    for i in range(len(energies)):
        if energies[i] > 0:
            unbound_index = i
            break


    if unbound_index == -1:
        print("All states are bound")
    else:
        scale = energies[1]-energies[0]
        print(f'Last bound state is v={unbound_index-1} with energy: {energies[unbound_index-1]} [J]')
        print(f'First unbound state is v={unbound_index} with energy: {energies[unbound_index]} [J]')
        plt.figure()

        plt.plot(r, U(r), color='black')
        plt.plot(r, ((states[unbound_index-1]*scale+ energies[unbound_index-1])))
        plt.plot(r, ((states[unbound_index]*scale+ energies[unbound_index])))
        plt.legend(["Morse Potential", "Last bound state", "First unbound state"])
        plt.grid()
        plt.xlabel(r'$r [m]$')
        plt.ylabel(r'$\Psi_v(r)$')
        plt.title(r'Wavefunctions of bound and unbound states')
        plt.xlim(0.5e-10, 0.5e-9)
        plt.ylim(-0.9e-18, 0.2e-18)


def partA_3(energies):
    energy = energies[1] - energies[0]
    print(f'Energy required for a transition from v=0 to v=1: {energy} [J]')
    print(f'hw_e = {(w)} [J]')


def partB_4_5(r, energies, states, m, title):
    m_energies = np.array(energies[:m])
    m_states = np.array(states[:][:m])
    print(title + f' Lowest {m} energies:')
    print(m_energies)
    plot_wavefuncs(r, m_energies, m_states, title)

def partB_6(energies_1, energies):
    # J=1, B_v = E_J/2
    E_J1 = energies_1-energies
    Bv = E_J1[:15] / 2

    coeff = np.array([[1, -0.5], [1, -1.5]])
    [B_e, alpha_e] = np.linalg.solve(coeff, Bv[:2])

    print(f'B_e: {B_e / (Planck*c)} [cm^-1]')
    print(f'alpha_e: {alpha_e / (Planck*c)} [cm^-1]')


def part_C(states_A, states_X):
    factors = np.zeros((13, 13))

    for i in range(0, 13):
        for j in range(0, 13):
            factors[i][j] = franck_condon(states_A[i], states_X[j])

    df = pd.DataFrame(factors)
    print(df)


w = 3737.76 * Planck * c  # [J]
wx = 84.881 * Planck * c  # [J]

r_e = 0.96966 * 1e-10  # [m]
D_e = (w**2)/(4*wx)  # [J]
a = w * np.sqrt(mu / (2*D_e)) / h_bar  # [m^-1]

# Part A
r = np.linspace(0.5e-10, 100e-10, 5000)
energies, states = get_eigenstates(T_matrix(r)+U_matrix(r))
partA_1(r, energies, states, 15)
partA_2(r, energies, states)
partA_2(r, energies, states)
_ = plt.xlim(0.5e-10, 0.2e-8)
_ = plt.ylim(-1e-20, 1e-20)
partA_3(energies)


# Part B
energies_1, states_1 = get_eigenstates(T_matrix(r) + U_matrix(r) + U_cent_matrix(r, 1))
energies_2, states_2 = get_eigenstates(T_matrix(r)+U_matrix(r)+U_cent_matrix(r, 2))
partB_4_5(r, energies_1, states_1, 15, "J=1")
_ = plt.xlim(0.5e-10, 5e-10)
_ = plt.ylim(-1e-18, 1e-19)
partB_4_5(r, energies_2, states_2, 15, "J=2")
_ = plt.xlim(0.5e-10, 5e-10)
_ = plt.ylim(-1e-18, 1e-19)
partB_6(energies_1, energies)


# Part C

# for A
w = 802.795 * Planck * c  # [J]
wx = 14.635 * Planck * c  # [J]
r_e = 1.514 * 1e-10  # [m]
D_e = (w**2)/(4*wx)  # [J]
mu = (m_O * m_O) / (m_O + m_O)  # [kg]
a = w * np.sqrt(mu / (2*D_e)) / h_bar  # [m^-1]

energies_A, states_A = get_eigenstates(T_matrix(r)+U_matrix(r))

# for X
w = 1580.3613 * Planck * c  # [J]
wx = 12.073 * Planck * c  # [J]
r_e = 1.207398 * 1e-10  # [m]
D_e = (w**2)/(4*wx)  # [J]
a = w * np.sqrt(mu / (2*D_e)) / h_bar  # [m^-1]

energies_X, states_X = get_eigenstates(T_matrix(r)+U_matrix(r))

part_C(states_A, states_X)