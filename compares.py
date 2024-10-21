import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# Simple square check
def squares_check(x1, x2):
    if not x1 or not x2:  # If its not empty
        return False
    
    # First
    min1 = min(x1)
    max1 = max(x1)

    # Second
    min2 = min(x2)
    max2 = max(x2)

    # Square usually near the center of canvas
    return (max1 - min1 <= 2) and (max2 - min2 <= 2)

# Main cyclic correlation
def cyclic_correlation(x1, x2):
    n = len(x1)
    max_corr = -1
    for shift in range(n):
        x2_shifted = np.roll(x2, shift)
        corr = np.corrcoef(x1, x2_shifted)[0, 1] 
        max_corr = max(max_corr, corr)
    return max_corr

# Main Pearsons correlation
def pearson_similarity(curve1, curve2):
    curve1_aligned, curve2_aligned = align_curves(curve1, curve2)
    # Вычисляем коэффициент корреляции Пирсона между кривыми
    similarity = pearsonr(curve1_aligned, curve2_aligned)[0]
  
    return similarity

# Interpolation of curves to the same len
def align_curves(curve1, curve2):
    length = max(len(curve1), len(curve2))
    x = np.linspace(0, 1, length)
    try:
        f1 = interp1d(np.linspace(0, 1, len(curve1)), curve1)
        f2 = interp1d(np.linspace(0, 1, len(curve2)), curve2)
        
        curve1_interp = f1(x)
        curve2_interp = f2(x)
        
        return curve1_interp, curve2_interp
    except ValueError as e:
        print(f"Interpolation error: {e}")
        return None, None

# Form-factors comparing
def check_ff_difference(ff1, ff2):
    average = (ff1 + ff2) / 2
    threshold = 0.1 * average
    absolute_difference = abs(ff1 - ff2)
    if absolute_difference > threshold:
        return False 
    else:
        return True

# Secondary cyclic correlation without zeros
def curve_similarity(curve1, curve2):
    # Deleting zeros
    curve1_no_zeros = [x for x in curve1 if x != 0]
    curve2_no_zeros = [x for x in curve2 if x != 0]

    curve1_i, curve2_i = align_curves(curve1_no_zeros, curve2_no_zeros)
    curve1_interp, curve2_interp = smooth_data_lists(curve1_i, curve2_i,3)
    
    cyclic_corr = cyclic_correlation(curve1_interp,curve2_interp)

    return cyclic_corr


def smooth_data_lists(data1, data2, method=1, polynomial_order=2):
  """
  Smoothes two lists of numbers using the selected method.

  Args:
  data1: The first list of numbers.
  data2: The second list of numbers.
  method: The number of the smoothing method:
  1 - Moving average (default)
  2 - Median filter
  3 - Savitzky-Golay filter
  polynomial_order: The order of the polynomial (used for method 3).

  Returns:
  A tuple of two lists: (smoothed_data1, smoothed_data2)
  """

  def moving_average(data):
    n = len(data)
    smoothed = []
    for i in range(n):
      start = max(0, i - 1) 
      end = min(n, i + 2)
      smoothed.append(sum(data[start:end]) / len(data[start:end]))
    return smoothed

  def median_filter(data):
    n = len(data)
    smoothed = []
    for i in range(n):
      start = max(0, i - 1)
      end = min(n, i + 2)
      smoothed.append(np.median(data[start:end]))
    return smoothed

  if method == 1:
    smoothed_data1 = moving_average(data1)
    smoothed_data2 = moving_average(data2)
  elif method == 2:
    smoothed_data1 = median_filter(data1)
    smoothed_data2 = median_filter(data2)
  elif method == 3:
    # For Savitzky-Golay we use the len(list) as a window size
    window_size = min(len(data1), len(data2))
    if window_size % 2 == 0:
      window_size -= 1  # Window must be odd
    smoothed_data1 = savgol_filter(data1, window_size, polynomial_order)
    smoothed_data2 = savgol_filter(data2, window_size, polynomial_order)
  else:
    raise ValueError("Wrong method. It should be: 1, 2, 3.")

  return smoothed_data1, smoothed_data2

