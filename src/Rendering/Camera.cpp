#include "Camera.h"
#include "Geometry.h"

Camera::Camera(vec3 pos, float heading, float pitch, float roll, float fov, float ncp, float fcp)
: m_idxInterpolNew(1),
  m_idxInterpolOld(0),
  m_lerpFactor(0.0f),
  m_forwardSpeed(0.f),
  m_strafeSpeed(0.0f),
  m_heading(heading),
  m_pitch(pitch),
  m_roll(roll),
  m_dir(0.0f, 0.0f, 0.0f),
  m_strafe(0.0f, 0.0f, 0.0f),
  m_up(0.0, 1.0, 0.0),
  m_tempDir(0.0, 0.0, 0.0),
  m_fov(fov), 
  m_ncp(ncp),
  m_fcp(fcp),
  m_frustum(NULL),
  m_renderFrustum(true),
  m_moveSpeed(0.01f),
  m_timer(0.0f),
  m_timerDiff(0.0f),
  m_secsPerFrame(0.0f),
  m_desiredDistance(4.0f),
  m_movementValue(0.0f),
  m_time(0.0f),
  m_cameraColor(1.0f, 1.0f, 1.0f),
  m_interpolate(false),
  //m_activeFrameSet(0),
  m_spline(NULL),
  m_splineView(NULL),
  m_splineSpeed(NULL),
  //m_activeFrameSetName("No Set Available"),
  m_aspect(1.3f),
  m_locked(false)
{
    m_frustum = new Frustum();
	m_showFrustum = new Frustum();	

    m_frustum->setCamInternals(m_fov, m_aspect, 0.2f, m_fcp);
	m_showFrustum->setCamInternals(m_fov, m_aspect, 0.2f, m_fcp*0.005f);

	m_spline = new Spline();
	m_splineView = new Spline();
	m_splineSpeed = new Spline();

    //!!!!!!!!!!!
    m_pos.x = pos.x;
    m_pos.y = pos.y;
    m_pos.z = -pos.z;

    update();

    m_hpTimer.reset();
    m_timer = (float)m_hpTimer.time();
}

Camera::~Camera()
{
    delete m_frustum;
	delete m_spline;
	delete m_splineView;
	delete m_splineSpeed;
}

void Camera::setPerspective(Transform &trans)
{    
    update();

    mat4 mat(m_matrix1);
    mat4 view;
    mat4 projection;

    view.set(m_matrix1);
    view = view.transpose();

    view *= mat4::translate(-m_pos.x, -m_pos.y, m_pos.z);
    m_aspect = params::inst()->window_x / (float)params::inst()->window_y;
    projection = mat4::perspective(m_fov, m_aspect, m_ncp, m_fcp);

    trans.projection = projection;
    trans.view = view;
    trans.viewProjection = projection * view;

    m_fixUp = normalize(vec3(view.a21, view.a22, view.a23));
    m_fixRight = normalize(vec3(view.a11, view.a12, view.a13));
    m_fixDir = normalize(cross(m_fixUp, m_fixRight));
    
}

void Camera::determineMovement()
{
    //Frame Independence
    m_timerDiff = (float)m_hpTimer.time() - m_timer;
    m_secsPerFrame = (float)(m_timerDiff);

    //m_secsPerFrameList.push_back(m_secsPerFrame);
    //if(m_secsPerFrameList.size() > 10)
    //    m_secsPerFrameList.erase(m_secsPerFrameList.end()-1);       

    m_movementValue = (float)(m_desiredDistance * m_secsPerFrame);
    m_timer = (float)m_hpTimer.time();
}

void Camera::update()
{
	//GLfloat Matrix[16];
	Quaternion q;

	// Make the Quaternions that will represent our rotations
    m_qPitch.fromAxisAngle(1.0f, 0.0f, 0.0f, m_pitch);
    m_qHeading.fromAxisAngle(0.0f, 1.0f, 0.0f, m_heading);
    m_qRoll.fromAxisAngle(0.0, 0.0f, 1.0f, m_roll);
	
	//m_pos.print();
	//std::cout << m_heading << m_pitch << m_roll << std::endl;
	// Combine the pitch and heading rotations and store the results in q
	q = m_qPitch * m_qHeading * m_qRoll;
    q.toMatrix(m_matrix1);
	
	// Create a matrix from the pitch Quaternion and get the j vector 
	// for our direction.
    m_qPitch.toMatrix(m_matrix2);
	m_dir.y = m_matrix2[9];    

	// Combine the heading and pitch rotations and make a matrix to get
	// the i and j vectors for our direction.
	q = m_qHeading * m_qPitch * m_qRoll;
    q.toMatrix(m_matrix2);
	m_dir.x = m_matrix2[8]; 
	m_dir.z = m_matrix2[10];

    float angle = m_heading * math_radians;

    float mY[9];
    mY[0] = cos(angle);  mY[1] = 0;  mY[2] = sin(angle);  
    mY[3] = 0;           mY[4] = 1;  mY[5] = 0;  
    mY[6] = -sin(angle); mY[7] = 0;  mY[8] = cos(angle);    

    //m_up = m_up.mulMatrix(mY);
    m_up = m_up.mulMatrix(mY);
    m_tempDir = m_dir;

	// Scale the direction by our speed.
    m_strafe = cross(m_dir, m_up) * m_strafeSpeed;
	m_dir = m_dir * m_forwardSpeed;        

    //m_dir.print();

    m_pos += m_dir;
    m_pos += m_strafe;
 
    m_realPos = vec3(m_pos.x, m_pos.y, -m_pos.z);
    
    vec3 fruDir = m_tempDir;
    fruDir.x *= -1;
    fruDir.y *= -1;

    m_frustum->setCamDef(m_realPos, fruDir, m_up);
	m_showFrustum->setCamDef(m_realPos, fruDir, m_up);

    m_forwardSpeed = 0.0f;
    m_strafeSpeed = 0.0f;
}

void Camera::changePitch(float degrees)
{
    if (!m_locked)
        m_pitch += degrees;
}

void Camera::changeHeading(float degrees)
{
    if (!m_locked)
        m_heading += degrees;
}

void Camera::changeRoll(float degrees)
{
    if (!m_locked) {
        m_roll += degrees;

        if (m_roll > 360.0f)
            m_roll -= 360.0f;
        else if (m_roll < -360.0f)
            m_roll += 360.0f;
    }
}

void Camera::moveForward(bool move)
{
    if (!m_locked) {
        if (move)
            m_forwardSpeed = m_movementValue;
        else
            m_forwardSpeed = 0.0f;
    }
}

void Camera::moveBackward(bool move)
{
    if (!m_locked) {
        if (move)
            m_forwardSpeed = -m_movementValue;
        else
            m_forwardSpeed = 0.0f;
    }
}

void Camera::strafeLeft(bool move)
{
    if (!m_locked) {
        if (move)
            m_strafeSpeed = m_movementValue;
        else
            m_strafeSpeed = 0.0f;
    }
}

void Camera::strafeRight(bool move)
{
    if (!m_locked) {
        if (move)
            m_strafeSpeed = -m_movementValue;
        else
            m_strafeSpeed = 0.0f;
    }
}

void Camera::increaseDistPerSec(float delta)
{
    m_desiredDistance += delta;
}

void Camera::decreaseDistPerSec(float delta)
{
    m_desiredDistance -= delta;
    if(m_desiredDistance < 0.1f)
        m_desiredDistance = 0.1f;
}

void Camera::setDistPerSec(float value)
{
    m_desiredDistance = value;
}

void Camera::render(const Transform &trans)
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    glPushClientAttrib(GL_CLIENT_ALL_ATTRIB_BITS);

    glEnableFixedFunction(trans);

        glDisable(GL_TEXTURE_2D);
        glColor4f(m_cameraColor.x, m_cameraColor.y, m_cameraColor.z, 1.0);

	    glPushMatrix();        
            glTranslatef(m_realPos.x, m_realPos.y, m_realPos.z);
            //glutSolidSphere(0.1, 10, 10);
        glPopMatrix();

        if(m_renderFrustum)		
            m_showFrustum->drawLines();

	    //if(m_showFrustum)		
	    //	m_showFrustum->drawPlanes();

        //renderFrames();
        //renderSpline();

    glDisableFixedFunction();

    glPopClientAttrib();
    glPopAttrib();
}

void Camera::renderFrames()
{ 
    glColor4f(m_cameraColor.x, m_cameraColor.y, m_cameraColor.z, 1.0f);

    glPushMatrix();
    glBegin(GL_LINE_STRIP);
    for(uint i=0; i<m_camFrames.size(); ++i)
    {              
        vec3 p = m_camFrames.at(i).pos;
        glVertex3f(p.x, p.y, -p.z);
    }
    glEnd();
    glPopMatrix();
}

void Camera::renderSpline()
{
    if(!m_spline || m_spline->numPoints() == 0)
        return;

    glPushMatrix();
    glPointSize(3.0f);
    glColor4f(0.0f, 0.0f, 0.0f, 1.0f);
    for(float f=0.0f; f<1; f+= 0.01f)
    {
        vec3 v = m_spline->interpolatedPoint(f);           

        glBegin(GL_POINTS);
            glVertex3f(v.x, v.y, v.z);
        glEnd();
    }
    glPopMatrix(); 
}

vec3 Camera::position() const
{
    return m_realPos;
}

float Camera::heading() const
{
    return m_heading;
}

float Camera::pitch() const
{
    return m_pitch;
}

float Camera::roll() const
{
    return m_roll;
}

vec3 Camera::direction() const
{
    vec3 dir = m_tempDir;
    dir.z *= -1;
    return dir;
}

vec3 Camera::up() const
{
    //return m_up;
    return m_fixUp;
}

vec3 Camera::right() const
{
    return m_fixRight;
}

Frustum *Camera::frustum()
{
    return m_frustum;
}

void Camera::updateCamInternals(float ratio)
{
	m_aspect = ratio;
    m_frustum->setCamInternals(m_fov, m_aspect, m_ncp, m_fcp);
	m_showFrustum->setCamInternals(m_fov, m_aspect, 0.2f, m_fcp*0.005f);
}

void Camera::addFrame(vec3 pos, float heading, float pitch, float desiredDistance)
{
    m_spline->addPoint(pos);

    vec3 viewDir(heading, pitch, 0.0f);
    m_splineView->addPoint(viewDir);

	vec3 speed(desiredDistance, 0.0f, 0.0f);
	m_splineSpeed->addPoint(speed);


    CameraFrame frame;
    frame.pos = pos;
    frame.pos.z *= -1;
    frame.headingDeg = heading;
    frame.pitchDeg = pitch;
	frame.desiredDistance = desiredDistance;

    m_camFrames.push_back(frame);
}

void Camera::autoAddFrame()
{
    vec3 m_tmpPos = m_pos;
    m_tmpPos.z *= -1.0f;
    m_spline->addPoint(m_tmpPos);

    vec3 viewDir(m_heading, m_pitch, 0.0f);
    m_splineView->addPoint(viewDir);

	vec3 speed(m_desiredDistance, 0.0f, 0.0f);
	m_splineSpeed->addPoint(speed);


    CameraFrame frame;
    frame.pos = m_pos;
    //frame.pos.z *= -1;
    frame.headingDeg = m_heading;
    frame.pitchDeg = m_pitch;
	frame.desiredDistance = speed.x;

    m_camFrames.push_back(frame);
}

void Camera::splineInterpolation()
{	
	float speed = m_movementValue;

	if(m_spline->numPoints() == 0)
		return;

	if(speed < 0.0001f)
		speed = 0.0001f;

	if(m_time > 1.0f)
	{
		m_time = 0.0f;
	}

	m_time += speed * 0.01f;

	vec3 v = m_spline->interpolatedPoint(m_time);
	vec3 d = m_splineView->interpolatedPoint(m_time);
	vec3 s = m_splineSpeed->interpolatedPoint(m_time);

	v.z *= -1;
	m_pos = v;
	m_heading = d.x;
	m_pitch = d.y;
	//m_desiredDistance = s.x;
}

void Camera::interpolate(float speed)
{
    if(m_camFrames.size() == 0)
        return;

    if(m_idxInterpolOld >= m_camFrames.size())
        m_idxInterpolOld = 0;

    if(m_idxInterpolNew >= m_camFrames.size())
        m_idxInterpolNew = 0;

    if(m_lerpFactor > 1.0)
        m_lerpFactor = 1.0;

    vec3 oldPos = m_camFrames.at(m_idxInterpolOld).pos;
    vec3 newPos = m_camFrames.at(m_idxInterpolNew).pos;
    vec3 curPos = vec3(0.0, 0.0, 0.0);

    curPos.x = hermiteInterpolation(oldPos.x/4.0f, oldPos.x, newPos.x, newPos.x/2.0f, m_lerpFactor, 1.0f, 1.0f);
    curPos.y = hermiteInterpolation(oldPos.y/4.0f, oldPos.y, newPos.y, newPos.y/2.0f, m_lerpFactor, 1.0f, 1.0f);
    curPos.z = hermiteInterpolation(oldPos.z/4.0f, oldPos.z, newPos.z, newPos.z/2.0f, m_lerpFactor, 1.0f, 1.0f);  
    
    m_pos = curPos;

    float oldHeadDeg = m_camFrames.at(m_idxInterpolOld).headingDeg;
    float newHeadDeg = m_camFrames.at(m_idxInterpolNew).headingDeg;
    float curHeadDeg = 0.0f;

    curHeadDeg = hermiteInterpolation(oldHeadDeg/4.0f, oldHeadDeg, newHeadDeg, newHeadDeg/2.0f, m_lerpFactor, 1.0f, 1.0f);

    m_heading = curHeadDeg;

    float oldPitchDeg = m_camFrames.at(m_idxInterpolOld).pitchDeg;
    float newPitchDeg = m_camFrames.at(m_idxInterpolNew).pitchDeg;
    float curPitchDeg = 0.0f;

    curPitchDeg = hermiteInterpolation(oldPitchDeg/4.0f, oldPitchDeg, newPitchDeg, newPitchDeg/2.0f, m_lerpFactor, 1.0f, 1.0f);

    m_pitch = curPitchDeg;


    m_lerpFactor += speed;

    if(curPos == newPos)
    {
        m_idxInterpolOld = m_idxInterpolNew;
        m_idxInterpolNew ++;
        m_lerpFactor = 0.0;
    }

    //setPerspective();
    //update();
}

void Camera::clearFrames()
{
    m_camFrames.clear();
    m_spline->clear();
    m_splineView->clear();
	m_splineSpeed->clear();
}

//void Camera::saveFrames(QString filePath)
//{
//
//
//     QFile file(filePath);
//     if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
//         return;
//
//     QTextStream out(&file);
//     
//     for(uint i=0; i<m_camFrames.size(); ++i)
//     {
//         CameraFrame cf = m_camFrames.at(i);
//         
//         if(i > 0)
//            out << "\n"; 
//         
//         out << cf.pos.x << " " << cf.pos.y << " " << cf.pos.z << " " << cf.headingDeg << " " << cf.pitchDeg << " " << cf.desiredDistance;                      
//     }
//
//     file.close();
//}
//
//void Camera::saveFrames()
//{
//	//QString filePath = m_frameSetFolder + "/" + m_activeFrameSetName;
//
//    QDate date = QDate::currentDate(); 
//    QTime time = QTime::currentTime();
//
//    int year = date.year();
//    int month = date.month();
//    int day = date.day();
//
//    int hour = time.hour();
//    int minute = time.minute();
//    int second = time.second();
//
//    QString sYear   = QString::number(year);
//    QString sMonth  = QString::number(month);
//    QString sDay    = QString::number(day);
//    QString sHour   = QString::number(hour);
//    QString sMinute = QString::number(minute);
//    QString sSecond = QString::number(second);
//    QString sNull   = QString::number(0);
//
//    QString fMonth  = month < 10 ? sNull + sMonth : sMonth;
//    QString fDay    = day < 10 ? sNull + sDay : sDay;
//    QString fHour   = hour < 10 ? sNull + sHour : sHour;
//    QString fMinute = minute < 10 ? sNull + sMinute : sMinute;
//    QString fSecond = second < 10 ? sNull + sSecond : sSecond;
//
//    QString fileName = sYear + fMonth + fDay + "_" + fHour + fMinute + fSecond + ".cam";
//
//    QString filePath = m_frameSetFolder + "/" + fileName;
// 	saveFrames(filePath);
//}
//
//void Camera::loadFrames(QString filePath)
//{
//     QFile file(filePath);
//     if (!file.open(QIODevice::ReadOnly | QIODevice::Text))
//     {
//         std::cout << "Camera::loadFrames() -> No Camera File available" << std::endl;
//         return;
//     }
//
//	 //m_spline->clear();
//	 //m_spline->clear();
//	 //m_spline->clear();
//
//	 delete m_spline;
//	 m_spline = NULL;
//
//	 delete m_splineView;
//	 m_splineView = NULL;
//
//	 delete m_splineSpeed;
//	 m_splineSpeed = NULL;
//
//	 m_spline = new Spline();
//	 m_splineView = new Spline();
//	 m_splineSpeed = new Spline();
//
//
//	 m_camFrames.clear();
//	 m_time = 0;
//
//     QTextStream in(&file);
//     
//     while (!in.atEnd()) 
//     {
//         vec3 v;
//         float heading = 0.0f;
//         float pitch = 0.0f;
//		 float desiredDistance = 0.0f;
//
//         in >> v.x >> v.y >> v.z >> heading >> pitch >> desiredDistance;
//         v.z *= -1;
//         addFrame(v, heading, pitch, desiredDistance);
//     }
//
//     file.close();
//}
//
//void Camera::loadFrameDirectory(QString folderPath)
//{
//	m_frameSetFolder = folderPath;
//
//	QDir dir(folderPath);
//	m_frameFileNames = dir.entryInfoList(QDir::Files);
//
//	if(m_frameFileNames.size())
//	{
//		QFileInfo fi = m_frameFileNames.at(0);
//
//		if(fi.exists())
//		{
//			loadFrames(fi.absoluteFilePath());
//			m_activeFrameSetName = fi.baseName() + "." + fi.suffix();
//		}
//	}	
//}
//
//void Camera::changeFrameSet()
//{
//	m_activeFrameSet ++;
//
//	if(m_activeFrameSet >= m_frameFileNames.size())
//		m_activeFrameSet = 0;
//
//	if(m_frameFileNames.size() > 0)
//	{
//		QFileInfo fi = m_frameFileNames.at(m_activeFrameSet);
//
//		if(fi.exists())
//		{
//			loadFrames(fi.absoluteFilePath());
//			m_activeFrameSetName = fi.baseName() + "." + fi.suffix();
//		}
//	}
//}
//
//void Camera::loadFramesFromProcedure()
//{
//    float radiusStep = 5.0 / (math_pi / 0.1);
//    float radius = 12.0f;
//    float alpha = math_pi;
//    float stepAlpha = 0.1f;
//
//    float pitch = 0.0f;
//    float stepPitch = 50.0f / (math_pi/0.1);
//
//    float posY = 3;
//    float stepY = 14.0f / (math_pi/0.1);
//
//    for(float phi = -math_pi; phi<=math_pi+0.01; phi += 0.1)
//    {
//        float headingDeg = (alpha - math_pi/2)*math_degrees;
//        float pitchDeg = pitch;
//        float desiredDistance = 2.0f;
//
//        vec3 pos;
//        pos.y = posY;
//
//        pos.x = cos(phi) * radius;
//        pos.z = sin(phi) * radius;        
//
//        alpha += stepAlpha;
//        //pitch += stepPitch;
//        //posY = 5;//stepY;
//        //radius += radiusStep;
//
//        addFrame(pos, headingDeg, pitchDeg, desiredDistance);
//    }
//
//    m_activeFrameSetName = "Radius 35";
//}
//
//QString Camera::frameSetName() const
//{
//	return m_activeFrameSetName;
//}

void Camera::setPosition(float x, float y, float z)
{
    m_pos.x = x;
    m_pos.y = y;
    m_pos.z = -z;
}

void Camera::setHeading(float degrees)
{
	m_heading = degrees;
}

void Camera::setPitch(float degrees)
{
	m_pitch = degrees;
}

void Camera::setColor(float r, float g, float b)
{
    m_cameraColor.x = r;
    m_cameraColor.y = g;
    m_cameraColor.z = b;
}

void Camera::setSpeed(float s)
{
	m_desiredDistance = s;
}

void Camera::toggleInterpolation()
{
    m_interpolate = !m_interpolate;
}

void Camera::interpolate()
{
    if(m_interpolate)
        splineInterpolation();
}

float Camera::fov() const
{
	return m_fov;
}

float Camera::ncp() const
{
	return m_ncp;
}

float Camera::fcp() const
{
	return m_fcp;
}

void Camera::lock()
{
    m_locked = !m_locked;
}