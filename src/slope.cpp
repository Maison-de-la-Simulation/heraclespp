#include "slope.hpp"

Slope::Slope(
        double const Uim1,
        double const Ui,
        double const Uip1)
{
        diffR = Uip1 - Ui; // Right slope
        diffL = Ui - Uim1; // Left
        R = diffR / diffL;
};

double Slope::VanLeer()
{
        if(diffL * diffR <= 0)
        {
            return 0;
        }
        else
        {
            return (1. / 2) * (diffR + diffL) * (4 * R) / ((R + 1) * (R + 1));
        }
}

double Slope::Minmod()
{
        if(diffL * diffR <= 0)
        {
            return 0;
        }
        else
        {
            return 1. / ((1. / diffL) + (1. / diffR));
        }
}

double Slope::VanAlbada()
{
        if(diffL * diffR <= 0)
        {
            return 0;
        }
        else
        {
            return (1. / 2) * (diffR + diffL) * (2 * R) / (R * R + 1);
        }
}
