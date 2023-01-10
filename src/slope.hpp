/**
 * @file slope.hpp
 * Slope in i
 */
#pragma once

//! Slope formule
//! @param[in] Uim1 float U_{i-1}^{n}
//! @param[in] Ui float U_{i}^{n}
//! @param[in] Uip1 float U_{i+1}^{n}
//! @return slope
class Slope
{
private :
      double diffl;
      double diffr;
      double R;

public :
      Slope(
          double const Uim1,
          double const Ui,
          double const Uip1);
      double VanLeer();
      double Minmod();
      double VanAlbada();
};
