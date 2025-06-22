// For Logger
#include "Logger.h"

// For python binding
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

// General includes
#include <iostream>
#include <fstream>
#include <boost/format.hpp>

#include <string>
#include <vector>
#include <tuple>
#include <cmath>
#include <variant>

// for CGAL::Nef_polyhedron_2
#include <CGAL/Exact_integer.h>
#include <CGAL/Filtered_extended_homogeneous.h>
#include <CGAL/Nef_polyhedron_2.h>

// For CGAL::Polyline_simplification_2
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_with_holes_2.h>
#include <CGAL/Polyline_simplification_2/simplify.h>
#include <CGAL/IO/WKT.h>

// Namespace & Type Definitions
namespace py = pybind11;
using namespace pybind11::literals;

// For CGAL::Nef_polyhedron_2
// Define the kernel and Nef polyhedron types
typedef CGAL::Exact_integer RT;
typedef CGAL::Filtered_extended_homogeneous<RT> Extended_kernel;
typedef CGAL::Nef_polyhedron_2<Extended_kernel> Nef_polyhedron;
typedef Extended_kernel::Standard_ray_2 Ray;
 
typedef Nef_polyhedron::Point Point;
typedef Nef_polyhedron::Line  Line;
typedef Nef_polyhedron::Explorer Explorer;

typedef Explorer::Face_const_iterator Face_const_iterator;
typedef Explorer::Hole_const_iterator Hole_const_iterator;
typedef Explorer::Halfedge_around_face_const_circulator Halfedge_around_face_const_circulator;
typedef Explorer::Vertex_const_handle Vertex_const_handle;

// For CGAL::Polyline_simplification_2
namespace PS = CGAL::Polyline_simplification_2;

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Polygon_2<K>                   Polygon_2;
typedef CGAL::Polygon_with_holes_2<K>        Polygon_with_holes_2;
typedef PS::Stop_above_cost_threshold        Stop;
typedef PS::Squared_distance_cost            Cost;


// Define the Class for Finding Support Size
// This class is responsible for processing halfplane intersections and calculating the support size
class HalfplaneIntersectionProcessor {
public:
    // Constructor
    HalfplaneIntersectionProcessor(int scene_size, double cost_threshold)
        : sceneSize(scene_size), costThreshold(cost_threshold), logger(nullptr) {}

    // Initiate logger
    void set_logger(Logger* log_ptr) {
        logger = log_ptr;
    }

    py::array_t<int> get_support(py::array_t<double> lines_3d) {
        auto arr = lines_3d.unchecked<3>();
        if (arr.ndim() != 3 || arr.shape(2) != 3) {
            throw std::runtime_error(
                "Input must be a 3‐D array with shape (T, N, 3)");
        }

        ssize_t T = arr.shape(0);
        ssize_t N = arr.shape(1);

        // prepare output length‐T
        py::array_t<int> out({ T });
        auto out_mut = out.mutable_unchecked<1>();

        for (ssize_t t = 0; t < T; ++t) {
            std::vector<Line> lines;
            lines.reserve(N+4);

            // bound constraints our area of interest
            lines.emplace_back( 1.0,  0.0,  sceneSize/2.0);
            lines.emplace_back(-1.0,  0.0,  sceneSize/2.0);
            lines.emplace_back( 0.0,  1.0,  sceneSize/2.0);
            lines.emplace_back( 0.0, -1.0,  sceneSize/2.0);

            // build lines for time‐slice t
            for (ssize_t i = 0; i < N; ++i) {
                // ax + by + c >= 0 (2 decimal place precision)
                int a = static_cast<int>(std::round(arr(t, i, 0) * 100.0));
                int b = static_cast<int>(std::round(arr(t, i, 1) * 100.0));
                int c = static_cast<int>(std::round(arr(t, i, 2) * 100.0));

                // NaN handling
                if (a == 0 && b == 0 && c == 0) {
                    continue;
                }
                
                lines.emplace_back(Line(a * 100, b * 100, c * 100));
            }

            Nef_polyhedron intersection_poly = intersect_halfspaces(lines);
            auto poly_vertices = explore(intersection_poly);
            auto simplified_vertices = simplify_polygon(poly_vertices);
            int support = count_edges(simplified_vertices);

            out_mut(t) = support;
        }

        return out; // output array shape: [T,]
    }

private:
    // 2D-point using double type
    struct DTuple {
        double x, y;
        DTuple(double _x, double _y): x(_x), y(_y) {}
    };
    // Constants
    int sceneSize;
    double costThreshold;
    Logger* logger;
    double tol = 1e-4;

    // Logger
    void log(const std::string& msg) const {
        if (logger) logger->log(msg);
    }
    
    // Intersection of half-spaces
    Nef_polyhedron intersect_halfspaces(const std::vector<Line>& lines) {
        Nef_polyhedron result(Nef_polyhedron::COMPLETE);
        for (size_t i = 0; i < lines.size(); ++i) {
            const Line& l = lines[i];

            Nef_polyhedron hs(l, Nef_polyhedron::INCLUDED);
            result *= hs;
    
            if (result.is_empty()) {
                throw std::runtime_error ("Intersection of halfspaces is empty!");
            }
        }
        return result;
    }

    // Explore Nef Polygon
    std::vector<DTuple> explore(const Nef_polyhedron& poly) {
        log("Explore Intersection Polygon: ");

        std::vector<DTuple> poly_vertices;
        std::ostringstream oss;

        Explorer explorer = poly.explorer();
        int i = 0;
        for(Face_const_iterator fit = explorer.faces_begin(); fit != explorer.faces_end(); ++fit, i++){
            if (!explorer.mark(fit)) {
                oss << "Face " << i << " is not marked" << std::endl;
                continue;
            }

            oss << "Face " << i << " is marked" << std::endl;
            // explore the outer face cycle if it exists
            Halfedge_around_face_const_circulator hafc = explorer.face_cycle(fit);
            if(hafc == Halfedge_around_face_const_circulator()){
                oss << "* has no outer face cycle" << std::endl;
            } else {
                oss << "* outer face cycle" << std::endl;
                oss << "  - halfedges around the face: " << std::endl;
                Halfedge_around_face_const_circulator done(hafc);
                do {
                    char c = (explorer.is_frame_edge(hafc))?'f':'e';
                    oss << c;
                    ++hafc;
                }while (hafc != done);
                oss << " ( f = frame edge, e = ordinary edge)" << std::endl;

                // We only need outer vertices of marked faces
                oss << "  - vertices around the face: " << std::endl;
                do {
                    Vertex_const_handle vh = explorer.target(hafc);
                    if (explorer.is_standard(vh)){
                        Point pt = explorer.point(vh);

                        double px = CGAL::to_double(pt.x());
                        double py = CGAL::to_double(pt.y());
                        oss << "      Point: " << "(" << px << "," << py << ")" << std::endl;

                        poly_vertices.emplace_back(px, py); // add to polygon vertices vector
                    } else {
                        oss << "      Ray: " << explorer.ray(vh);
                        throw std::runtime_error ("We never encounter a Ray on account of our bounded frame!");
                    }
                    ++hafc;
                }while (hafc != done);
            }

            // explore the holes if the face has holes
            Hole_const_iterator hit = explorer.holes_begin(fit), end = explorer.holes_end(fit);
            if(hit == end){
                oss << "* has no holes" << std::endl;
            } else {
                oss << "* has holes" << std::endl;
                for(; hit != end; hit++){
                    Halfedge_around_face_const_circulator hafc(hit), done(hit);
                    oss << "  - halfedges around the hole: " << std::endl;
                    do {
                        char c = (explorer.is_frame_edge(hafc))?'f':'e';
                        oss << c;
                        ++hafc;
                    }while (hafc != done);
                    oss << " ( f = frame edge, e = ordinary edge)" << std::endl;
                }
            }
        }
        oss << "done";
        log(oss.str());

        return poly_vertices;
    }
    
    // simplify polygon (merge small neighboring edges)
    std::vector<DTuple> simplify_polygon(const std::vector<DTuple>& poly_vertices) {
        // Convert to CGAL::Polygon_2
        Polygon_2 polygon;

        for (const auto& point : poly_vertices) {
            polygon.push_back(K::Point_2(point.x, point.y));
        }

        if (! polygon.is_simple()) {
            throw std::runtime_error ("simplify_polygon: Polgon edges intersect!");
        } else {
            // Simplify the polygon
            Cost cost;

            if (polygon.size() > 2) {
                polygon = PS::simplify(polygon, cost, Stop(costThreshold));
            }

            std::vector<DTuple> simplified_vertices;

            std::ostringstream oss;
            oss << "Simplified Polygon Vertices:";
            for (auto it = polygon.vertices_begin(); it != polygon.vertices_end(); ++it) {
                double x = it->x();
                double y = it->y();
                simplified_vertices.emplace_back(x, y);
                oss << " (" << x << ", " << y << ") ;";
            }
            log(oss.str());

            if (simplified_vertices.size()>2) {
                simplified_vertices.emplace_back(
                    simplified_vertices[0].x, simplified_vertices[0].y
                ); // Close the polygon
            }

            return simplified_vertices;
        }
    }

    // calculate support from edges
    int count_edges(const std::vector<DTuple>& simplified_vertices) {
        int support_size = 0;
    
        for (size_t i = 0; i < simplified_vertices.size() - 1; ++i) {
            size_t j = i + 1;

            double x1 = simplified_vertices[i].x;
            double y1 = simplified_vertices[i].y;
            double x2 = simplified_vertices[j].x;
            double y2 = simplified_vertices[j].y;

            if (std::abs(x1 - x2) < tol &&
                std::abs(std::abs(x1) - sceneSize/2) < tol) {
                continue; // skip vertical lines at the edges
            }
            else if (std::abs(y1 - y2) < tol &&
                std::abs(std::abs(y1) - sceneSize/2) < tol) {
                continue; // skip horizontal lines at the edges
            } else {
                support_size += 1;
            }
        }
        log("Calculated Support Size: " + std::to_string(support_size) + "\n");
        return support_size;
    }
};

// create python binding
PYBIND11_MODULE(halfplane_module, m) {
    py::class_<Logger>(m, "Logger")
        .def(py::init<const std::string&, bool>(),
            "log_file"_a,
             "enabled"_a = false
        )
        .def("log", &Logger::log)
        .def("enable", &Logger::enable);

    py::class_<HalfplaneIntersectionProcessor>(m, "HalfplaneIntersectionProcessor")
        .def(py::init<int, double>(),
            "scene_size"_a,
            "tolerance"_a
        )
        .def("set_logger", &HalfplaneIntersectionProcessor::set_logger, py::return_value_policy::reference)
        .def("get_support", &HalfplaneIntersectionProcessor::get_support,
            "Compute support size for each time slice in a (T,N,3) constraints array");
}