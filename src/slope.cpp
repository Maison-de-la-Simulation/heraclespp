#include "slope.hpp"

Slope::Slope(
        double const Uim1,
        double const Ui,
        double const Uip1)
{
        diffl = Uip1 - Ui;
        diffr = Ui - Uim1;
        R = diffl / diffr;
};

double Slope::VanLeer()
{
        double slopeChoice;
        if(diffl * diffr <= 0)
        {
          slopeChoice = 0;
        }
        else
        {
          slopeChoice = (1. / 2) * (diffr + diffl) * (4 * R) / ((R + 1) * (R + 1));
        }
        return slopeChoice;

}

double Slope::Minmod()
{
        double slopeChoice;
        if(diffl * diffr <= 0)
        {
          slopeChoice = 0;
        }
        else
        {
          double slopeChoice = 1. / ((1. / diffl) + (1. / diffr));
        }
        return slopeChoice;
}

double Slope::VanAlbada()
{
        double slopeChoice;
        if(diffl * diffr <= 0)
        {
          slopeChoice = 0;
        }
        else
        {
          double slopeChoice  = (1. / 2) * (diffr + diffl) * (2 * R) / (R * R + 1);
        }
        return slopeChoice;
}
