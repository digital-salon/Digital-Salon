#ifndef SCENE_H
#define SCENE_H


#include "imgui/imgui.h"
#include "imgui/imgui_impl_glfw.h"
#include "imgui/imgui_impl_opengl3.h"
//#include <ppl.h>
#include <filesystem>


#include "Rendering/Mesh2D.h"
#include "solver.h"
#include "HairGen.h"

#include "Perm.h"
#include "Copilot.h"
#include "AIRender.h"

class NiceGrid;
class Light;
class VertexBufferObject;
class Shader;
class CameraManager;
class Object;
class Texture;

// A structure to hold individual chat messages
struct ChatMessage {
    std::string text;
};

class Scene
{
public: 
    Scene(CameraManager* camManager, const HairInfo& hairInfo, const HeadInfo& headInfo, ExpOpt opt = NONE);
    ~Scene();

    // Engine Callables
    void EmbeddedGUI();
    void UpdateEmbedded();

    void ResetAll(std::string hairPath);

    // Others
    void UpdateSimulation();
    void ResetSimulation();
	void Update(Transform &trans, float delta);
    void SceneDrawGUI();
    void InitSimulation();
    void InitHead(const HeadInfo& headInfo, const bool& objSeq);
    void RenderWorld(const Transform &trans);  
    void RenderObjects(const Transform &trans);  
    void RenderObjectsDepth(const Transform &trans);
    void Reset();

    vector<Light*> m_lights;

private:

    // Embedding
    Copilot mCopilot;
    Perm mPerm;
    AIRender mAIRender;

    // Experiments
    void SetupExperiment(ExpOpt opt);
    bool ExportSpaceExploration = false;
    void ExportExplorationLG();
    void ExportExplorationLS();
    //void SmallExporter(string fname);

    // Simulation
    shared_ptr<Solver> SimSolver;
    bool Generated = false;

    // Generation
    float3 HairColor = make_float3(0.82, 0.42, 0.24);
    shared_ptr<HairGenerator> HairGen;
    HairInfo HairData;
    HeadInfo HeadData;
    void GenerateHair();
    void SaveTriangleSelection();
    void LoadTriangleSelection();

    // Solid Objects
    vector<shared_ptr<SimpleObject>> Meshes;

    // Eulerian grid
    void GenerateHairGrid();

    // Bones
    shared_ptr<VertexBufferObject> VboBone;
    shared_ptr<SimpleObject> BoneMesh;

    // Rendering
    NiceGrid *m_niceGrid;
    CameraManager *m_cameraManager;
    vector<shared_ptr<VertexBufferObject>> VboMeshes;
    vector<shared_ptr<VertexBufferObject>> VboHair;
    shared_ptr<VertexBufferObject> VboGridBox;

    // VBO builders
    void BuildVboHairGrid(const bool& update = false);
    void BuildVboHair(vector<vector<HairNode>>& strands);
    void BuildVboMeshes();
    void BuildVboBone();
    void BuildSimVBO();

    // Helpers
    void InitHairCut(Transform& trans);
    void SelectTriangles(Transform& trans, int val);
    //void InitHairCut(Transform& trans);
    //void UpdateHairGrab(Transform& trans);
    //void UpdateGrabPosition(Transform& trans);
    void SelectMesh(int val);

    // Misc helpers
    void SetupExportFolder();
    //bool LastGrabState = false;

    std::vector<ChatMessage> messages;
    bool scrollToBottom = false;
    void AddMessage(const std::string& message) {
        messages.push_back({message});
    }

    // Copilot
    bool is_simulation_init=false;

    
    GLuint LoadImage(const char* filename);

    GLuint image1, image2, image3;
    bool showImageSelection=false;
    int selectedImage=-1;



    //AIRender
    bool showImageWindow=false;
    GLuint generated_image;

    void SaveOpenGLImage(const char* filename, int width, int height);


    // temp misc
    bool temp = false;


};




#endif

 