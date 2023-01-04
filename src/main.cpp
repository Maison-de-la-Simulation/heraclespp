#include <iostream>
#include <Kokkos_Core.hpp>

int main(int argc, char** argv)
{
    Kokkos::ScopeGuard guard;
    std::cout << "Hello world\n";
    return 0;
}
