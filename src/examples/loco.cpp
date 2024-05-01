#include <srbd/srbd.hpp>

using namespace srbd;

int main()
{
    auto ANYmal = SingleRigidBodyDynamics::create_robot(RobotType::ANYmal);
    return 0;
}
